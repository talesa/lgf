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


class OrthogonalSylvesterBijection(Bijection):
    def __init__(self, num_input_channels, num_ortho_vecs=None, diag_activation='tanh', orthogonalization_steps=100):
        shape = (num_input_channels,)
        super().__init__(x_shape=shape, z_shape=shape)

        self.r1_triu = nn.Parameter(num_input_channels, num_input_channels)
        self.r2_triu = nn.Parameter(num_input_channels, num_input_channels)
        self.r1_unrestricted_diag = nn.Parameter(num_input_channels)
        self.r2_unrestricted_diag = nn.Parameter(num_input_channels)
        self.q_parameters = nn.Parameter(num_input_channels, num_ortho_vecs)
        self.b = nn.Parameter(1, 1, num_ortho_vecs)

        # self.num_input_channels = num_input_channels
        # self.num_ortho_vecs = num_ortho_vecs

        self.r_diag_activation = activation_functions[diag_activation]

        # Orthogonalization parameters
        self.orthogonalization_steps = orthogonalization_steps

        if self.num_ortho_vecs == self.z_size:
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        identity = identity.unsqueeze(0)

        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=0)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()

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
        r1[:, self.diag_idx, self.diag_idx] = diag_r1

        r2 = self.r2_triu * self.triu_mask
        diag_r2 = self.r_diag_activation(self.r2_unrestricted_diag)
        r2[:, self.diag_idx, self.diag_idx] = diag_r2

        # Create orthogonal matrices
        q_ortho = self.construct_orthogonal_matrix(self.q_parameters)

        qr1 = torch.bmm(q_ortho, r1)
        qr2 = torch.bmm(q_ortho, r2.transpose(2, 1))

        z = x
        r2qzb = torch.bmm(z, qr2.unsqueeze(0)) + self.b
        z = z + torch.bmm(self.h(r2qzb), qr1.transpose(2, 1).unsqueeze(0))
        # z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j = 1. + diag_j
        log_det_j = diag_j.abs().log()

        return {
            "z": z,
            "log-jac": log_det_j.view(x.shape[0], 1)
        }

    def construct_orthogonal_matrix(self, q):
        """
        Construct orthogonal matrix from its parameterization.
        :param q:  q contains batches of matrices, shape : (1, z_size * num_ortho_vecs)
        :return: orthogonalized matrix, shape: (1, z_size, num_ortho_vecs)
        """

        # Reshape to shape (1, z_size * num_ortho_vecs)
        q = q.view(1, self.z_size * self.num_ortho_vecs)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.view(dim0, self.z_size, self.num_ortho_vecs)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
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
        amat = amat.view(-1, self.num_flows, self.z_size, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)

        return amat

    def h(self, x):
        return torch.nn.functional.tanh(x)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _z_to_x(self, z, **kwargs):
        # This could be implemented using fixed-point iteration method.
        raise NotImplementedError("_z_to_x not implemented for the Sylvester flow.")
