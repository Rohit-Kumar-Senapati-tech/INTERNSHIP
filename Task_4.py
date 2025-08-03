# Production Optimization using Linear Programming
# Business Problem: Manufacturing Company Product Mix Optimization

"""
BUSINESS PROBLEM SCENARIO:
TechManufacturing Inc. produces three types of electronic devices:
- Smartphones (S)
- Tablets (T) 
- Laptops (L)

The company wants to maximize profit while considering:
- Limited production capacity
- Material constraints
- Labor hour limitations
- Market demand constraints

OBJECTIVE: Determine the optimal production mix to maximize profit
"""

# Import required libraries
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

print("="*60)
print("PRODUCTION OPTIMIZATION USING LINEAR PROGRAMMING")
print("="*60)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# STEP 1: PROBLEM DEFINITION AND DATA SETUP
# =============================================================================

print("STEP 1: PROBLEM DEFINITION AND DATA SETUP")
print("-" * 50)

# Product data
products = ['Smartphones', 'Tablets', 'Laptops']
product_codes = ['S', 'T', 'L']

# Profit per unit (in dollars)
profit = {
    'S': 200,  # Smartphone profit
    'T': 300,  # Tablet profit  
    'L': 500   # Laptop profit
}

# Resource requirements per unit
# Assembly hours required per product
assembly_hours = {
    'S': 2,   # 2 hours per smartphone
    'T': 3,   # 3 hours per tablet
    'L': 4    # 4 hours per laptop
}

# Material cost per unit (as constraint)
material_units = {
    'S': 1,   # 1 unit of material per smartphone
    'T': 2,   # 2 units of material per tablet
    'L': 3    # 3 units of material per laptop
}

# Testing hours required per product
testing_hours = {
    'S': 1,   # 1 hour testing per smartphone
    'T': 1.5, # 1.5 hours testing per tablet
    'L': 2    # 2 hours testing per laptop
}

# Available resources (constraints)
max_assembly_hours = 1000    # Maximum assembly hours available
max_material_units = 800     # Maximum material units available
max_testing_hours = 600      # Maximum testing hours available

# Market demand constraints (maximum units that can be sold)
max_demand = {
    'S': 300,  # Max 300 smartphones can be sold
    'T': 200,  # Max 200 tablets can be sold
    'L': 150   # Max 150 laptops can be sold
}

# Minimum production requirements (contracts)
min_production = {
    'S': 50,   # Must produce at least 50 smartphones
    'T': 30,   # Must produce at least 30 tablets
    'L': 20    # Must produce at least 20 laptops
}

# Display problem data
print("PRODUCT PROFITABILITY:")
for product, code in zip(products, product_codes):
    print(f"  {product}: ${profit[code]} profit per unit")

print("\nRESOURCE REQUIREMENTS PER UNIT:")
print("Product      Assembly(hrs)  Material(units)  Testing(hrs)")
print("-" * 55)
for product, code in zip(products, product_codes):
    print(f"{product:<12} {assembly_hours[code]:<13} {material_units[code]:<15} {testing_hours[code]}")

print(f"\nAVAILABLE RESOURCES:")
print(f"  Assembly Hours: {max_assembly_hours}")
print(f"  Material Units: {max_material_units}")
print(f"  Testing Hours: {max_testing_hours}")

print(f"\nMARKET CONSTRAINTS:")
for product, code in zip(products, product_codes):
    print(f"  {product}: Min {min_production[code]}, Max {max_demand[code]} units")

# =============================================================================
# STEP 2: LINEAR PROGRAMMING MODEL FORMULATION
# =============================================================================

print("\n" + "="*60)
print("STEP 2: LINEAR PROGRAMMING MODEL FORMULATION")
print("-" * 50)

# Create the linear programming problem
prob = pulp.LpProblem("Production_Optimization", pulp.LpMaximize)

# Decision variables: number of units to produce for each product
x_s = pulp.LpVariable("Smartphones", lowBound=0, cat='Integer')
x_t = pulp.LpVariable("Tablets", lowBound=0, cat='Integer')
x_l = pulp.LpVariable("Laptops", lowBound=0, cat='Integer')

# Store variables in a dictionary for easier access
variables = {'S': x_s, 'T': x_t, 'L': x_l}

print("DECISION VARIABLES:")
print("  x_s = Number of Smartphones to produce")
print("  x_t = Number of Tablets to produce") 
print("  x_l = Number of Laptops to produce")

# Objective function: Maximize profit
prob += (profit['S'] * x_s + profit['T'] * x_t + profit['L'] * x_l), "Total_Profit"

print(f"\nOBJECTIVE FUNCTION:")
print(f"  Maximize: {profit['S']}*x_s + {profit['T']}*x_t + {profit['L']}*x_l")

print(f"\nCONSTRAINTS:")

# Constraint 1: Assembly hours
prob += (assembly_hours['S'] * x_s + assembly_hours['T'] * x_t + 
         assembly_hours['L'] * x_l <= max_assembly_hours), "Assembly_Hours"
print(f"  Assembly: {assembly_hours['S']}*x_s + {assembly_hours['T']}*x_t + {assembly_hours['L']}*x_l ≤ {max_assembly_hours}")

# Constraint 2: Material units
prob += (material_units['S'] * x_s + material_units['T'] * x_t + 
         material_units['L'] * x_l <= max_material_units), "Material_Units"
print(f"  Material: {material_units['S']}*x_s + {material_units['T']}*x_t + {material_units['L']}*x_l ≤ {max_material_units}")

# Constraint 3: Testing hours
prob += (testing_hours['S'] * x_s + testing_hours['T'] * x_t + 
         testing_hours['L'] * x_l <= max_testing_hours), "Testing_Hours"
print(f"  Testing: {testing_hours['S']}*x_s + {testing_hours['T']}*x_t + {testing_hours['L']}*x_l ≤ {max_testing_hours}")

# Constraint 4: Market demand (upper bounds)
prob += x_s <= max_demand['S'], "Max_Smartphones"
prob += x_t <= max_demand['T'], "Max_Tablets" 
prob += x_l <= max_demand['L'], "Max_Laptops"
print(f"  Market Demand: x_s ≤ {max_demand['S']}, x_t ≤ {max_demand['T']}, x_l ≤ {max_demand['L']}")

# Constraint 5: Minimum production requirements
prob += x_s >= min_production['S'], "Min_Smartphones"
prob += x_t >= min_production['T'], "Min_Tablets"
prob += x_l >= min_production['L'], "Min_Laptops"
print(f"  Min Production: x_s ≥ {min_production['S']}, x_t ≥ {min_production['T']}, x_l ≥ {min_production['L']}")

# =============================================================================
# STEP 3: SOLVE THE OPTIMIZATION PROBLEM
# =============================================================================

print("\n" + "="*60)
print("STEP 3: SOLVING THE OPTIMIZATION PROBLEM")
print("-" * 50)

# Solve the problem
print("Solving the linear programming problem...")
prob.solve()

# Check the solution status
status = pulp.LpStatus[prob.status]
print(f"Solution Status: {status}")

if status == 'Optimal':
    print("✅ Optimal solution found!")
else:
    print("❌ No optimal solution found. Check constraints.")

# =============================================================================
# STEP 4: EXTRACT AND DISPLAY RESULTS
# =============================================================================

print("\n" + "="*60)
print("STEP 4: OPTIMIZATION RESULTS")
print("-" * 50)

if status == 'Optimal':
    # Extract optimal values
    optimal_production = {}
    optimal_production['S'] = int(x_s.varValue)
    optimal_production['T'] = int(x_t.varValue)
    optimal_production['L'] = int(x_l.varValue)
    
    # Calculate total profit
    total_profit = pulp.value(prob.objective)
    
    print("OPTIMAL PRODUCTION PLAN:")
    print("-" * 30)
    for product, code in zip(products, product_codes):
        units = optimal_production[code]
        unit_profit = profit[code]
        total_product_profit = units * unit_profit
        print(f"  {product:<12}: {units:>3} units (${total_product_profit:,} profit)")
    
    print(f"\nTOTAL MAXIMUM PROFIT: ${total_profit:,.2f}")
    
    # =============================================================================
    # STEP 5: RESOURCE UTILIZATION ANALYSIS
    # =============================================================================
    
    print("\n" + "="*60)
    print("STEP 5: RESOURCE UTILIZATION ANALYSIS")
    print("-" * 50)
    
    # Calculate resource usage
    assembly_used = sum(assembly_hours[code] * optimal_production[code] for code in product_codes)
    material_used = sum(material_units[code] * optimal_production[code] for code in product_codes)
    testing_used = sum(testing_hours[code] * optimal_production[code] for code in product_codes)
    
    # Calculate utilization percentages
    assembly_util = (assembly_used / max_assembly_hours) * 100
    material_util = (material_used / max_material_units) * 100
    testing_util = (testing_used / max_testing_hours) * 100
    
    print("RESOURCE UTILIZATION:")
    print("-" * 40)
    print(f"Assembly Hours: {assembly_used:>6.1f} / {max_assembly_hours} ({assembly_util:>5.1f}%)")
    print(f"Material Units: {material_used:>6.1f} / {max_material_units} ({material_util:>5.1f}%)")
    print(f"Testing Hours:  {testing_used:>6.1f} / {max_testing_hours} ({testing_util:>5.1f}%)")
    
    # Identify bottlenecks
    print(f"\nBOTTLENECK ANALYSIS:")
    utilization_rates = {
        'Assembly': assembly_util,
        'Material': material_util, 
        'Testing': testing_util
    }
    
    max_util_resource = max(utilization_rates, key=utilization_rates.get)
    max_util_rate = utilization_rates[max_util_resource]
    
    print(f"  Primary Bottleneck: {max_util_resource} ({max_util_rate:.1f}% utilized)")
    
    if max_util_rate >= 95:
        print(f"  ⚠️  {max_util_resource} is a critical constraint limiting production")
    elif max_util_rate >= 80:
        print(f"  ⚠️  {max_util_resource} is approaching capacity limits")
    else:
        print(f"  ✅ {max_util_resource} has available capacity for expansion")
    
    # =============================================================================
    # STEP 6: SENSITIVITY ANALYSIS
    # =============================================================================
    
    print("\n" + "="*60)
    print("STEP 6: SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    # Analyze shadow prices (dual values) for constraints
    print("SHADOW PRICES (Value of additional resources):")
    print("-" * 45)
    
    # Note: Shadow prices show the marginal value of relaxing each constraint
    constraints_info = {
        'Assembly_Hours': 'Additional assembly hour',
        'Material_Units': 'Additional material unit',
        'Testing_Hours': 'Additional testing hour'
    }
    
    for constraint_name in constraints_info:
        try:
            shadow_price = prob.constraints[constraint_name].pi
            if shadow_price is not None and shadow_price > 0:
                print(f"  {constraints_info[constraint_name]}: ${shadow_price:.2f}")
            else:
                print(f"  {constraints_info[constraint_name]}: $0.00 (non-binding)")
        except:
            print(f"  {constraints_info[constraint_name]}: Not available")
    
    # =============================================================================
    # STEP 7: BUSINESS INSIGHTS AND RECOMMENDATIONS
    # =============================================================================
    
    print("\n" + "="*60)
    print("STEP 7: BUSINESS INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    
    print("KEY INSIGHTS:")
    print("-" * 15)
    
    # Product mix insights
    total_units = sum(optimal_production.values())
    print(f"1. PRODUCTION MIX:")
    for product, code in zip(products, product_codes):
        percentage = (optimal_production[code] / total_units) * 100
        print(f"   • {product}: {percentage:.1f}% of total production")
    
    # Profitability insights
    print(f"\n2. PROFITABILITY:")
    profit_per_product = {}
    for code in product_codes:
        profit_per_product[code] = optimal_production[code] * profit[code]
    
    for product, code in zip(products, product_codes):
        contribution = (profit_per_product[code] / total_profit) * 100
        print(f"   • {product}: {contribution:.1f}% of total profit")
    
    # Constraint analysis
    print(f"\n3. CONSTRAINT ANALYSIS:")
    if assembly_util > 90:
        print("   • Assembly capacity is nearly fully utilized - consider expansion")
    if material_util > 90:
        print("   • Material supply is nearly exhausted - secure additional suppliers")
    if testing_util > 90:
        print("   • Testing capacity is constrained - invest in testing equipment")
    
    print(f"\n4. RECOMMENDATIONS:")
    print("   • Focus marketing efforts on the most profitable product mix")
    print(f"   • Consider increasing capacity for {max_util_resource.lower()} resources")
    print("   • Evaluate possibility of increasing prices for high-demand products")
    print("   • Monitor market demand changes and adjust production accordingly")
    
    # =============================================================================
    # STEP 8: CREATE SUMMARY DATAFRAME
    # =============================================================================
    
    print("\n" + "="*60)
    print("STEP 8: SUMMARY TABLE")
    print("-" * 50)
    
    # Create comprehensive summary
    summary_data = []
    for product, code in zip(products, product_codes):
        summary_data.append({
            'Product': product,
            'Optimal_Production': optimal_production[code],
            'Unit_Profit': profit[code],
            'Total_Profit': optimal_production[code] * profit[code],
            'Assembly_Hours_Used': optimal_production[code] * assembly_hours[code],
            'Material_Units_Used': optimal_production[code] * material_units[code],
            'Testing_Hours_Used': optimal_production[code] * testing_hours[code],
            'Min_Requirement_Met': 'Yes' if optimal_production[code] >= min_production[code] else 'No',
            'Below_Max_Demand': 'Yes' if optimal_production[code] <= max_demand[code] else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Resource summary
    print(f"\nRESOURCE SUMMARY:")
    resource_summary = pd.DataFrame({
        'Resource': ['Assembly Hours', 'Material Units', 'Testing Hours'],
        'Used': [assembly_used, material_used, testing_used],
        'Available': [max_assembly_hours, max_material_units, max_testing_hours],
        'Utilization_%': [assembly_util, material_util, testing_util],
        'Remaining': [max_assembly_hours - assembly_used, 
                     max_material_units - material_used,
                     max_testing_hours - testing_used]
    })
    print(resource_summary.to_string(index=False))

    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Maximum achievable profit: ${total_profit:,.2f}")
    print(f"Optimal solution determined for {total_units} total units")
    print("This solution maximizes profit while respecting all constraints.")

else:
    print("No feasible solution found. Please check constraint compatibility.")

# =============================================================================
# ADDITIONAL NOTES FOR IMPLEMENTATION
# =============================================================================

print(f"\n" + "="*60)
print("IMPLEMENTATION NOTES")
print("-" * 50)
print("1. This model can be extended to include:")
print("   • Seasonal demand variations")
print("   • Multiple production facilities")
print("   • Inventory holding costs")
print("   • Setup costs for production changes")
print("   • Quality constraints")
print()
print("2. Real-world considerations:")
print("   • Update demand forecasts regularly")
print("   • Monitor actual resource consumption vs. estimates")
print("   • Consider supply chain disruptions")
print("   • Validate profit assumptions")
print()
print("3. Model validation:")
print("   • Test with historical data")
print("   • Perform what-if analysis")
print("   • Compare with current production plans")

print(f"\nNotebook completed successfully!")
print(f"Save this analysis for business decision-making reference.")