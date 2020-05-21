import torch
from torch import nn

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
    def __init__(self, num_input_channels, q_parameters_nelem, diag_activation='tanh', num_ortho_vecs=None):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        if num_ortho_vecs is None:
            num_ortho_vecs = num_input_channels

        self.r1_triu = nn.Parameter(torch.zeros(num_ortho_vecs, num_ortho_vecs))
        self.r2_triu = nn.Parameter(torch.zeros(num_ortho_vecs, num_ortho_vecs))
        self.r1_unrestricted_diag = nn.Parameter(torch.zeros(num_ortho_vecs))
        self.r2_unrestricted_diag = nn.Parameter(torch.zeros(num_ortho_vecs))
        self.q_parameters = nn.Parameter(torch.zeros(q_parameters_nelem))
        self.b = nn.Parameter(torch.zeros(num_ortho_vecs))

        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

        self.num_input_channels = num_input_channels
        self.num_ortho_vecs = num_ortho_vecs

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

        # Create orthogonal matrices
        q_ortho = self.construct_orthogonal_matrix(self.q_parameters)

        qr1 = torch.matmul(q_ortho, r1)
        qr2 = torch.matmul(q_ortho, r2.transpose(0, 1))

        z = x
        r2qzb = torch.matmul(z, qr2) + self.b.unsqueeze(0)
        z = z + torch.matmul(self.h(r2qzb), qr1.transpose(1, 0))

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
        q = q.view(1, self.num_input_channels * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.view(dim0, self.num_input_channels, self.num_ortho_vecs)

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
        amat = amat.view(self.num_input_channels, self.num_ortho_vecs)

        return amat


class HouseholderSylvesterBijection(BaseSylvesterBijection):
    def __init__(self, num_input_channels, num_householder, diag_activation='tanh'):
        super().__init__(num_input_channels, diag_activation=diag_activation, num_ortho_vecs=num_input_channels,
                         q_parameters_nelem=num_input_channels*num_householder)

        self.num_householder = num_householder
        assert self.num_householder > 0

        self.cond = 1.e-6

    def construct_orthogonal_matrix(self, q):
        """
        Construct an orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (z_size * num_householder)
        :return: orthogonalized matrix, shape: (z_size, z_size)
        """

        # Reshape to shape (num_flows * num_householder, z_size)
        q = q.view(-1, self.num_input_channels)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)   # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = self._eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # Reshaping: first dimension is num_flows
        amat = amat.view(-1, self.num_householder, self.num_input_channels, self.num_input_channels)

        tmp = amat[:, 0]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.num_input_channels, self.num_input_channels)

        amat = amat.view(self.num_input_channels, self.num_input_channels)

        return amat
