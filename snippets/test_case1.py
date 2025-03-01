import unittest
from shopping_cart import Product, Cart, Order

# ==============================
# Test A
# ==============================
class TestAShoppingCart(unittest.TestCase):
    def test_end_to_end_order_process(self):
        # Create actual objects from the shopping_cart system
        cart = Cart()
        product1 = Product(1, "Widget", 10.0)
        product2 = Product(2, "Gadget", 20.0)
        
        # Simulate user actions: adding products to the cart.
        cart.add_product(product1, 2)  # Total: 2 * 10.0 = 20.0
        cart.add_product(product2, 1)  # Total: 1 * 20.0 = 20.0
        # Combined total should be 40.0
        
        # Create an order using the actual cart
        order = Order(cart, "Alice")
        result = order.process_order()
        
        # Verify the order processed successfully and the total amount is correct.
        self.assertTrue(result)
        self.assertEqual(order.status, "Processed")
        self.assertAlmostEqual(order.total_amount, 40.0)
