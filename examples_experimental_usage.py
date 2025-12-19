"""
Example: Using FlagGems Experimental Operators

This example demonstrates how to use experimental operators in FlagGems.
"""

import torch
import flag_gems.experimental as fg_exp


def example_direct_call():
    """Example 1: Direct call without enable()"""
    print("=" * 60)
    print("Example 1: Direct Call")
    print("=" * 60)

    from flag_gems.experimental.generated.pointwise import huber_loss

    # Create test tensors
    input = torch.randn(10, 10, device="cuda", dtype=torch.float32)
    target = torch.randn(10, 10, device="cuda", dtype=torch.float32)

    # Call experimental operator directly
    result = huber_loss(input, target, reduction=1, delta=1.0)
    print(f"Result shape: {result.shape}")
    print(f"Result: {result.item():.6f}")


def example_enable_global():
    """Example 2: Enable experimental operators globally"""
    print("\n" + "=" * 60)
    print("Example 2: Global Enable")
    print("=" * 60)

    # Enable experimental operators
    fg_exp.enable()

    # Create test tensors
    input = torch.randn(10, 10, device="cuda", dtype=torch.float32)
    target = torch.randn(10, 10, device="cuda", dtype=torch.float32)

    # Now torch will use experimental implementation
    result = torch.ops.aten.huber_loss(input, target, 1, 1.0)
    print(f"Result shape: {result.shape}")
    print(f"Result: {result.item():.6f}")

    # Check what's enabled
    print(f"\nEnabled operators: {fg_exp.list_enabled_ops()}")

    # Disable when done
    fg_exp.disable()


def example_selective_enable():
    """Example 3: Selectively enable operator groups"""
    print("\n" + "=" * 60)
    print("Example 3: Selective Enable")
    print("=" * 60)

    # Enable only generated operators
    print("Enabling only 'generated' operators:")
    fg_exp.enable(groups=['generated'])
    print(f"Enabled: {fg_exp.list_enabled_ops()}")
    fg_exp.disable()

    # Enable only custom operators
    print("\nEnabling only 'custom' operators:")
    fg_exp.enable(groups=['custom'])
    print(f"Enabled: {fg_exp.list_enabled_ops()}")
    fg_exp.disable()

    # Enable all
    print("\nEnabling all operators:")
    fg_exp.enable(groups=None)
    print(f"Enabled: {fg_exp.list_enabled_ops()}")
    fg_exp.disable()


def example_context_usage():
    """Example 4: Using in context"""
    print("\n" + "=" * 60)
    print("Example 4: Context Usage")
    print("=" * 60)

    input = torch.randn(10, 10, device="cuda", dtype=torch.float32)
    target = torch.randn(10, 10, device="cuda", dtype=torch.float32)

    # Use experimental operators in a specific context
    fg_exp.enable()
    try:
        result = torch.ops.aten.huber_loss(input, target, 1, 1.0)
        print(f"Result: {result.item():.6f}")
    finally:
        fg_exp.disable()

    print("Experimental operators disabled after use")


if __name__ == "__main__":
    print("\n🧪 FlagGems Experimental Operators Examples\n")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. These examples require a CUDA device.")
        exit(1)

    try:
        # Run examples
        example_direct_call()
        example_enable_global()
        example_selective_enable()
        example_context_usage()

        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
