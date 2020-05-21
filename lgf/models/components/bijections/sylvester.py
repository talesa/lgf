import torch
from torch import nn

import numpy as np

from .bijection import Bijection

# Models below are based on https://github.com/riannevdberg/sylvester-flows

# TODO in the future
#  first implement one where everything is done one by one
# These Bijection classes are able to perform multiple steps of the same Bijection what breaks the intuitive API
#  of the rest of the codebase.
#  The reason for this is that this way the somewhat costly orthogonalization procedure can be performed for all the
#  instances of the

activation_functions = {'tanh': nn.Tanh(), 'softplus': nn.Softplus()}


class BaseSylvesterBijection(Bijection):
    def __init__(self, num_input_channels, q_parameters_nelem, diag_activation='tanh', num_ortho_vecs=None,
                 permute=None):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        if num_ortho_vecs is None:
            num_ortho_vecs = num_input_channels

        self.r1_triu = nn.Parameter(torch.zeros(num_ortho_vecs, num_ortho_vecs))
        self.r2_triu = nn.Parameter(torch.zeros(num_ortho_vecs, num_ortho_vecs))
        self.r1_unrestricted_diag = nn.Parameter(torch.zeros(num_ortho_vecs))
        self.r2_unrestricted_diag = nn.Parameter(torch.zeros(num_ortho_vecs))
        if permute is None:
            self.q_parameters = nn.Parameter(torch.zeros(q_parameters_nelem))
        self.b = nn.Parameter(torch.zeros(num_ortho_vecs))

        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

        self.num_input_channels = num_input_channels
        self.z_size = num_input_channels
        self.num_ortho_vecs = num_ortho_vecs
        self.permute = permute

        self.r_diag_activation = activation_functions[diag_activation]

        identity = torch.eye(num_ortho_vecs, num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        # Put tensors in buffer so that they will be moved to GPU if needed by any call of .cuda()
        self.register_buffer('_eye', identity)
        self._eye.requires_grad = False
        self.register_buffer('triu_mask', triu_mask)
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def _x_to_z(self, x, **kwargs):
        """
        Conditions on diagonals of R1 and R2 for invertibility are enforced within this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param x: shape: (batch_size, num_input_channels)
        :return: z, log-jac
        """

        # Enforce constraint on the R1, R2 diagonals
        # Save r diags for log_det_j
        # TODO this might throw in-place operation error that might affect backprop
        #  if this doesn't work create a separate parameter just for the diagonal
        r1 = self.r1_triu * self.triu_mask
        diag_r1 = self.r_diag_activation(self.r1_unrestricted_diag)
        r1[self.diag_idx, self.diag_idx] = diag_r1

        r2 = self.r2_triu * self.triu_mask
        diag_r2 = self.r_diag_activation(self.r2_unrestricted_diag)
        r2[self.diag_idx, self.diag_idx] = diag_r2

        if self.permute is None:
            # Create orthogonal matrices
            q_ortho = self.construct_orthogonal_matrix(self.q_parameters)
            qr1 = torch.matmul(q_ortho, r1)
            qr2 = torch.matmul(q_ortho, r2.transpose(0, 1))
            z = x
        elif self.permute is True:
            qr1 = r1
            qr2 = r2
            z = x[:, self.permutation]
        elif self.permute is False:
            qr1 = r1
            qr2 = r2
            z = x

        # TODO there might be a bug here?
        r2qzb = torch.matmul(z, qr2) + self.b.unsqueeze(0)
        z_delta = torch.matmul(self.h(r2qzb), qr1.transpose(1, 0))

        if self.permute is True:
            z_delta = z_delta[:, self.permutation]

        z = x + z_delta

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb) * diag_j.unsqueeze(0)
        diag_j = 1. + diag_j
        log_det_j = diag_j.abs().log().sum(-1)

        return {
            "z": z,
            "log-jac": log_det_j.view(x.shape[0], 1)
        }

    def construct_orthogonal_matrix(self, q):
        """
        Construct orthogonal matrix from its parameterization.
        :param q: q contains batches of matrix parameters (as required by a particular parameterization).
        :return: orthogonalized matrix, shape: (num_input_channels, num_ortho_vecs)
        """
        raise NotImplementedError

    def h(self, x):
        return torch.tanh(x)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _z_to_x(self, z, **kwargs):
        # This could be implemented using fixed-point iteration method.
        raise NotImplementedError("_z_to_x not implemented for the Sylvester flow.")


class OrthogonalSylvesterBijection(BaseSylvesterBijection):
    def __init__(self, num_input_channels, diag_activation='tanh', num_ortho_vecs=None, orthogonalization_steps=100):
        super().__init__(num_input_channels, diag_activation=diag_activation, num_ortho_vecs=num_ortho_vecs,
                         q_parameters_nelem=num_input_channels*num_ortho_vecs)

        # Orthogonalization procedure parameters
        self.orthogonalization_steps = orthogonalization_steps

        if num_ortho_vecs == num_input_channels:
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

    def construct_orthogonal_matrix(self, q):
        """
        Construct orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (1, num_input_channels * num_ortho_vecs)
        :return: orthogonalized matrix, shape: (1, num_input_channels, num_ortho_vecs)
        """

        # Reshape to shape (1, z_size * num_ortho_vecs)
        q = q.view(1, self.z_size * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.view(dim0, self.z_size, self.num_ortho_vecs)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.orthogonalization_steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).item()
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            raise Exception('Orthogonalization not complete.')
            # print('\nWARNING WARNING WARNING: orthogonalization not complete')
            # print('\t Final max norm =', max_norm)
            #
            # print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(self.z_size, self.num_ortho_vecs)

        return amat


class HouseholderSylvesterBijection(BaseSylvesterBijection):
    def __init__(self, num_input_channels, num_householder, diag_activation='tanh'):
        super().__init__(num_input_channels, diag_activation=diag_activation,
                         q_parameters_nelem=num_input_channels*num_householder)

        self.num_householder = num_householder
        assert self.num_householder > 0

    def construct_orthogonal_matrix(self, q):
        """
        Construct an orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (z_size * num_householder)
        :return: orthogonalized matrix, shape: (z_size, z_size)
        """

        # Reshape to shape (num_flows * num_householder, z_size)
        q = q.view(-1, self.z_size)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # Reshaping: first dimension is num_flows
        amat = amat.view(-1, self.num_householder, self.z_size, self.z_size)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.z_size, self.z_size)

        amat = amat.view(self.z_size, self.z_size)

        return amat


class TriangularSylvesterBijection(BaseSylvesterBijection):
    """
    Alternates between setting the orthogonal matrix equal to permutation and identity matrix for each flow.
    """
    def __init__(self, num_input_channels, diag_activation='tanh', permute=True):
        super().__init__(num_input_channels, diag_activation=diag_activation, q_parameters_nelem=None, permute=permute)

        if permute:
            permutation = torch.arange(num_input_channels - 1, -1, -1).long()
            self.register_buffer('permutation', permutation)


class ExponentialSylvesterBijection(BaseSylvesterBijection):
    def __init__(self, num_input_channels, diag_activation='tanh'):
        super().__init__(num_input_channels, diag_activation=diag_activation,
                         q_parameters_nelem=num_input_channels*num_input_channels)

    def construct_orthogonal_matrix(self, q):
        """
        Construct an orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (z_size * z_size)
        :return: orthogonalized matrix, shape: (z_size, z_size)
        """

        A = q.view(-1, self.z_size, self.z_size)
        A = A.triu(1)
        A = A - A.transpose(-1, -2)
        B = exp_skew(A)

        B = B.view(self.z_size, self.z_size)

        return B


class CayleySylvesterBijection(BaseSylvesterBijection):
    def __init__(self, num_input_channels, diag_activation='tanh'):
        super().__init__(num_input_channels, diag_activation=diag_activation,
                         q_parameters_nelem=num_input_channels*num_input_channels)

    def construct_orthogonal_matrix(self, q):
        """
        Construct an orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (z_size * z_size)
        :return: orthogonalized matrix, shape: (z_size, z_size)
        """

        A = q.view(-1, self.z_size, self.z_size)
        A = A.triu(1)
        A = A - A.transpose(-1, -2)
        B = self.cayley(A)

        B = B.view(self.z_size, self.z_size)

        return B

    @staticmethod
    def cayley(X):
        n = X.size(-1)
        Id = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0)
        return torch.solve(Id - X, Id + X)[0]


# Exponential utils
def p7(X):
    n = X.size(-1)
    Id = torch.eye(n, dtype=X.dtype, device=X.device).unsqueeze(0)
    X1 = torch.matmul(X, X)
    X2 = torch.matmul(X1, X1)
    X3 = torch.matmul(X1, X2)
    P1 = 17297280. *  Id + 1995840. * X1 + 25200. * X2 + 56. * X3
    P2 = torch.matmul(X,
                  8648640. * Id + 277200. * X1 + 1512. * X2 + X3)
    return P1 + P2, P1 - P2


def exp_pade(X):
    p7pos, p7neg = p7(X)
    return torch.solve(p7pos, p7neg)[0]


def matrix_pow_batch(A, k):
    ksorted, iidx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count)
    A = torch.matrix_power(A, 2**ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[iidx[processed:]] = torch.matrix_power(A[iidx[processed:]], 2**new.item())
        processed += count[exp]
    return A


def expm(X, exp=exp_pade):
    """
    Scaling and squaring trick
    """
    norm = torch.norm(X, dim=(1, 2))
    more = norm >= 1.
    k = torch.zeros(norm.size(), dtype=torch.long, device=X.device)
    k[more] = torch.ceil(torch.log2(norm[more])).long()
    # Terrible hack, make it cleaner in the future
    B = torch.pow(.5, k.float()).unsqueeze(1).unsqueeze(2).expand_as(X) * X
    E = exp(B)
    return matrix_pow_batch(E, k)


def expm_frechet(A, E):
    n = A.size(-1)
    M = torch.zeros(A.size(0), 2*n, 2*n, dtype=A.dtype, device=A.device, requires_grad=False)
    M[:, :n, :n] = A
    M[:, n:, n:] = A
    M[:, :n, n:] = E
    return expm(M)[:, :n, n:]


class exp_skew_class(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        B = expm(A)
        ctx.save_for_backward(A, B)
        return B

    @staticmethod
    def backward(ctx, G):
        def skew(X):
            return .5 * (X - X.transpose(1, 2))
        # print(G)
        A, B = ctx.saved_tensors
        grad = skew(B.transpose(1, 2).matmul(G))
        out = B.matmul(expm_frechet(-A, grad))
        # correct precission errors
        return skew(out)


def exp_taylor(X):
    n = X.size(0)
    Id = torch.eye(n, dtype=X.dtype, device=X.device)
    X2 = torch.mm(X, X)
    X4 = torch.mm(X2, X2)
    inv_fact = [1.]
    for i in range(1, 8):
        inv_fact.append(inv_fact[i-1] / float(i))
    return Id + inv_fact[4]*X4 + X2.mm(inv_fact[2]*Id + inv_fact[6]*X4) +\
           X.mm(Id + inv_fact[5]*X4 + X2.mm(inv_fact[3]*Id + inv_fact[7]*X4))


def expI(X, exp=exp_taylor):
    """
    Scaling and squaring trick
    """
    norm = X.norm()
    if norm < 1.:
        k = 0
        B = X
    else:
        k = int(np.ceil(np.log2(float(norm))))
        B = X * (2.**-k)
    E = exp(B)
    for _ in range(k):
        E = torch.mm(E, E)
    return E


exp_skew = exp_skew_class.apply
