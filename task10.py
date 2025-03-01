# public class DiscountCalculator {
#     private double dblOrderTotal;
#     private double dblDiscountRate;
#     private double dblFinalPrice;

#     public DiscountCalculator(double orderTotal, double discountRate) {
#         this.dblOrderTotal = orderTotal  
#         this.dblDiscountRate = discountRate;
#     }

#     public double CalculateFinalPrice() {
#         if(dblDiscountRate < 0 || dblDiscountRate > 100) {
#             throw new IllegalArgumentException("Invalid discount rate");
#         }
#         dblFinalPrice = dblOrderTotal - (dblOrderTotal * (dblDiscountRate / 100.0));
#         return dblFinalPrice;
#     }

#     @Override
#     public String toString() {
#         return "DiscountCalculator[OrderTotal=" + dblOrderTotal +
#                ", DiscountRate=" + dblDiscountRate +
#                ", FinalPrice=" + dblFinalPrice + "]";
#     }

#     public static void main(String[] args) {
#         DiscountCalculator calculator = new DiscountCalculator(200.0, 10.0);
#         double finalPrice = calculator.CalculateFinalPrice();
#         System.out.println("Final Price: $" + finalPrice);
#         System.out.println(calculator);
#     }
# }

# public class DiscountCalculator {
#     private double dblOrderTotal;
#     private double dblDiscountRate;
#     private double dblFinalPrice;

#     public DiscountCalculator(double orderTotal, double discountRate) {
#         this.dblOrderTotal = orderTotal;
#         this.dblDiscountRate = discountRate;
#     }

#     public double CalculateFinalPrice() {
#         if(dblDiscountRate < 0 || dblDiscountRate > 100) {
#             throw new IllegalArgumentException("Invalid discount rate");
#         }
#         // Bug: Using '+' instead of '-' to apply discount.
#         dblFinalPrice = dblOrderTotal + (dblOrderTotal * (dblDiscountRate / 100.0));
#         return dblFinalPrice;
#     }

#     @Override
#     public String toString() {
#         return "DiscountCalculator[OrderTotal=" + dblOrderTotal +
#                ", DiscountRate=" + dblDiscountRate +
#                ", FinalPrice=" + dblFinalPrice + "]";
#     }

#     public static void main(String[] args) {
#         DiscountCalculator calculator = new DiscountCalculator(200.0, 10.0);
#         double finalPrice = calculator.CalculateFinalPrice();
#         System.out.println("Final Price: $" + finalPrice);
#         System.out.println(calculator);
#     }
# }

# C:
# public class DiscountCalculator {
#     private double dblOrderTotal;     // Order total
#     private double dblDiscountRate;   // Discount rate in percentage
#     private double dblFinalPrice;     // Final price after discount

#     public DiscountCalculator(double orderTotal, double discountRate) {
#         this.dblOrderTotal = orderTotal;
#         this.dblDiscountRate = discountRate;
#     }

#     public double CalculateFinalPrice() {
#         if(dblDiscountRate < 0 || dblDiscountRate > 100) {
#             throw new IllegalArgumentException("Invalid discount rate");
#         }
#         dblFinalPrice = dblOrderTotal - (dblOrderTotal * (dblDiscountRate / 100.0));
#         return dblFinalPrice;
#     }

#     @Override
#     public String toString() {
#         return "DiscountCalculator[OrderTotal=" + dblOrderTotal +
#                ", DiscountRate=" + dblDiscountRate +
#                ", FinalPrice=" + dblFinalPrice + "]";
#     }

#     public static void main(String[] args) {
#         DiscountCalculator calculator = new DiscountCalculator(200.0, 10.0);
#         double finalPrice = calculator.CalculateFinalPrice();
#         System.out.println("Final Price: $" + finalPrice);
#         System.out.println(calculator);
#     }
# }

# D:
# public class DiscountCalculator {
#     private double dblOrderTotal;
#     private double dblDiscountRate;
#     private double dblFinalPrice;

#     public DiscountCalculator(double orderTotal, double discountRate) {
#         this.dblDiscountRate = discountRate;
#     }

#     public double CalculateFinalPrice() {
#         if(dblDiscountRate < 0 || dblDiscountRate > 100) {
#             throw new IllegalArgumentException("Invalid discount rate");
#         }
#         dblFinalPrice = dblOrderTotal - (dblOrderTotal * (dblDiscountRate / 100.0));
#         return dblFinalPrice;
#     }

#     @Override
#     public String toString() {
#         return "DiscountCalculator[OrderTotal=" + dblOrderTotal +
#                ", DiscountRate=" + dblDiscountRate +
#                ", FinalPrice=" + dblFinalPrice + "]";
#     }

#     public static void main(String[] args) {
#         DiscountCalculator calculator = new DiscountCalculator(200.0, 10.0);
#         double finalPrice = calculator.CalculateFinalPrice();
#         System.out.println("Final Price: $" + finalPrice);
#         System.out.println(calculator);
#     }
# }

# E: 
# public class DiscountCalculator {
#     private double dblOrderTotal;
#     private double dblDiscountRate;
#     private double dblFinalPrice;

#     public DiscountCalculator(double orderTotal, double discountRate) {
#         this.dblOrderTotal = orderTotal;
#         this.dblDiscountRate = discountRate;
#     }

#     public double CalculateFinalPrice() {
#         if(dblDiscountRate < 0 || dblDiscountRate > 100) {
#             throw new IllegalArgumentException("Invalid discount rate");
#         }
#         dblFinalPrice = dblOrderTotal - (dblOrderTotal * (dblDiscountRate / 100.0));
#         return dblFinalPrice;
#     }

#     @Override
#     public String toString() {
#         return "DiscountCalculator[OrderTotal=" + dblOrderTotal +
#                ", DiscountRate=" + dblDiscountRate +
#                ", FinalPrice=" + dblFinalPrice + "]";
#     }

#     public static void main(String[] args) {
#         DiscountCalculator calculator = new DiscountCalculator(200.0, 10.0);
#         double finalPrice = calculator.CalcFinalPrice();  
#         System.out.println("Final Price: $" + finalPrice);
#         System.out.println(calculator);
#     }
# }

# F:
# public class SimpleCalculator {
#     public double add(double dblA, double dblB) {
#         return dblA + dblB;
#     }

#     public double subtract(double dblA, double dblB) {
#         return dblA - dblB;
#     }

#     public double multiply(double dblA, double dblB) {
#         return dblA * dblB;
#     }

#     public double divide(double dblA, double dblB) {
#         if (dblB == 0) {
#             throw new IllegalArgumentException("Cannot divide by zero");
#         }
#         return dblA / dblB;
#     }

#     public static void main(String[] args) {
#         SimpleCalculator calc = new SimpleCalculator();
#         System.out.println("Add: " + calc.add(10, 5));
#         // Bug: Incorrect method name called ("substract" instead of "subtract").
#         System.out.println("Subtract: " + calc.substract(10, 5));  
#         System.out.println("Multiply: " + calc.multiply(10, 5));
#         System.out.println("Divide: " + calc.divide(10, 5));
#     }
# }

# G:
# public class SimpleCalculator {
#     public double add(double dblA, double dblB) {
#         return dblA + dblB;
#     }

#     public double subtract(double dblA, double dblB) {
#         double dblResult;  
#         return dblResult;  
#     }

#     public double multiply(double dblA, double dblB) {
#         return dblA * dblB;
#     }

#     public double divide(double dblA, double dblB) {
#         if (dblB == 0) {
#             throw new IllegalArgumentException("Cannot divide by zero");
#         }
#         return dblA / dblB;
#     }

#     public static void main(String[] args) {
#         SimpleCalculator calc = new SimpleCalculator();
#         System.out.println("Add: " + calc.add(10, 5));
#         System.out.println("Subtract: " + calc.subtract(10, 5)); 
#         System.out.println("Multiply: " + calc.multiply(10, 5));
#         System.out.println("Divide: " + calc.divide(10, 5));
#     }
# }

# H:
# public class SimpleCalculator {
#     public double add(double dblA, double dblB) {
#         return dblA + dblB;
#     }

#     public double subtract(double dblA, double dblB) {
#         return dblA - dblB;
#     }

#     public double multiply(double dblA, double dblB) {
#         return dblA + dblB;  // Logical error: should be multiplication.
#     }

#     public double divide(double dblA, double dblB) {
#         if (dblB == 0) {
#             throw new IllegalArgumentException("Cannot divide by zero");
#         }
#         return dblA / dblB;
#     }

#     public static void main(String[] args) {
#         SimpleCalculator calc = new SimpleCalculator();
#         System.out.println("Add: " + calc.add(10, 5));         
#         System.out.println("Subtract: " + calc.subtract(10, 5));   
#         System.out.println("Multiply: " + calc.multiply(10, 5));   
#         System.out.println("Divide: " + calc.divide(10, 5));       
#     }
# }

# I:
# public class SimpleCalculator {

#     public double add(double dblA, double dblB) {
#         return dblA + dblB;
#     }

#     public double subtract(double dblA, double dblB) {
#         return dblA - dblB;
#     }

#     public double multiply(double dblA, double dblB) {
#         return dblA * dblB;
#     }

#     public double divide(double dblA, double dblB) {
#         if (dblB == 0) {
#             throw new IllegalArgumentException("Cannot divide by zero");
#         }
#         return dblA / dblB;
#     }

#     public static void main(String[] args) {
#         SimpleCalculator calc = new SimpleCalculator();
#         System.out.println("Add: " + calc.add(10, 5));         
#         System.out.println("Subtract: " + calc.subtract(10, 5));   
#         System.out.println("Multiply: " + calc.multiply(10, 5));   
#         System.out.println("Divide: " + calc.divide(10, 5));       
#     }
# }

# J:
# public class SimpleCalculator {
#     public double add(double dblA, double dblB) {
#         return dblA + dblB  
#     }

#     public double subtract(double dblA, double dblB) {
#         return dblA - dblB;
#     }

#     public double multiply(double dblA, double dblB) {
#         return dblA * dblB;
#     }

#     public double divide(double dblA, double dblB) {
#         if (dblB == 0) {
#             throw new IllegalArgumentException("Cannot divide by zero");
#         }
#         return dblA / dblB;
#     }

#     public static void main(String[] args) {
#         SimpleCalculator calc = new SimpleCalculator();
#         System.out.println("Add: " + calc.add(10, 5));
#         System.out.println("Subtract: " + calc.subtract(10, 5));
#         System.out.println("Multiply: " + calc.multiply(10, 5));
#         System.out.println("Divide: " + calc.divide(10, 5));
#     }
# }

# A     Syntax Error Bug	Missing semicolon (;) at the end of the constructor line: this.dblOrderTotal = orderTotal
# B     OK Case	            No syntax or logical errors; correctly calculates the final price
# C     OK Case	            Proper implementation without errors
# D     Uninitialized Field Bug	dblOrderTotal is not initialized in the constructor, leading to potential runtime errors
# E     Method Call Error   Bug	Incorrect method name CalcFinalPrice() instead of CalculateFinalPrice()
# F     Method Call Error   Bug	Calls calc.substract() instead of calc.subtract(), which does not exist
# G     Logical Error       Bug	Variable dblResult is declared but not assigned before being returned in subtract() method
# H     OK Case	            Proper implementation, though multiplication logic could be checked manually
# I     OK Case	            No syntax or logical errors
# J     Syntax Error Bug	Missing semicolon (;) in the add() method before the return statement
