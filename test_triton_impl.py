import unittest

import torch

from triton_impl import DDPM as StreamingDDPM
from unfold_impl import DDPM as UnfoldDDPM


class TestStreamingDDPM(unittest.TestCase):
    def _models(self, channels: int, kernel_size: int, max_generated_bytes: int = 0):
        torch.manual_seed(123)
        reference = UnfoldDDPM(channels, kernel_size)
        candidate = StreamingDDPM(
            channels,
            kernel_size,
            spatial_tile_size=11,
            max_generated_bytes=max_generated_bytes,
            recompute=True,
        )
        candidate.load_state_dict(reference.state_dict())
        return reference, candidate

    def test_forward_matches_unfold_across_tile_boundaries(self):
        for kernel_size in (1, 3, 5):
            with self.subTest(kernel_size=kernel_size):
                reference, candidate = self._models(3, kernel_size)
                x = torch.randn(2, 3, 5, 7)
                y = torch.randn_like(x)
                with torch.no_grad():
                    expected = reference(x, y)
                    actual = candidate(x, y)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_backward_matches_unfold_with_recomputation(self):
        for budget in (0, 1024**3):
            with self.subTest(max_generated_bytes=budget):
                reference, candidate = self._models(4, 3, budget)
                x_ref = torch.randn(2, 4, 5, 7, requires_grad=True)
                y_ref = torch.randn_like(x_ref, requires_grad=True)
                x_new = x_ref.detach().clone().requires_grad_()
                y_new = y_ref.detach().clone().requires_grad_()

                expected = reference(x_ref, y_ref)
                actual = candidate(x_new, y_new)
                expected_grads = torch.autograd.grad(
                    expected.square().mean(), (x_ref, y_ref, *reference.parameters())
                )
                actual_grads = torch.autograd.grad(
                    actual.square().mean(), (x_new, y_new, *candidate.parameters())
                )

                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, atol=3e-6, rtol=3e-5
                    )

    def test_rejects_invalid_shapes(self):
        model = StreamingDDPM(4, 3)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            model(torch.randn(1, 4, 5, 5), torch.randn(1, 4, 4, 5))

    def test_rejects_even_kernel(self):
        with self.assertRaisesRegex(ValueError, "positive odd"):
            StreamingDDPM(4, 2)


if __name__ == "__main__":
    unittest.main()
