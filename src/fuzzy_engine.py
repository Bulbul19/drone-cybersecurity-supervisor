import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Inputs (Antecedents) ---
satellites = ctrl.Antecedent(np.arange(0, 16, 1), 'satellites')
hdop = ctrl.Antecedent(np.arange(0, 11, 0.1), 'hdop')
vibration = ctrl.Antecedent(np.arange(0, 51, 1), 'vibration')

# --- Output (Consequent) ---
gps_trust = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'gps_trust_score')

# --- Membership Functions ---
satellites['FEW'] = fuzz.trimf(satellites.universe, [0, 0, 6])
satellites['OKAY'] = fuzz.trimf(satellites.universe, [4, 8, 12])
satellites['MANY'] = fuzz.trimf(satellites.universe, [10, 15, 15])

hdop['EXCELLENT'] = fuzz.trimf(hdop.universe, [0, 1, 1.7])
hdop['GOOD'] = fuzz.trimf(hdop.universe, [1.5, 2.5, 3.5])
hdop['POOR'] = fuzz.trimf(hdop.universe, [3, 10, 10])

vibration['LOW'] = fuzz.trimf(vibration.universe, [0, 5, 15])
vibration['MEDIUM'] = fuzz.trimf(vibration.universe, [10, 25, 40])
vibration['HIGH'] = fuzz.trimf(vibration.universe, [35, 50, 50])

gps_trust['VERY_LOW'] = fuzz.trimf(gps_trust.universe, [0, 0.1, 0.25])
gps_trust['LOW'] = fuzz.trimf(gps_trust.universe, [0.15, 0.35, 0.55])
gps_trust['MEDIUM'] = fuzz.trimf(gps_trust.universe, [0.45, 0.6, 0.75])
gps_trust['HIGH'] = fuzz.trimf(gps_trust.universe, [0.65, 0.8, 0.9])
gps_trust['VERY_HIGH'] = fuzz.trimf(gps_trust.universe, [0.85, 1, 1])

# --- Rule Base for 3 Inputs ---
rule1 = ctrl.Rule(satellites['FEW'] | hdop['POOR'], gps_trust['VERY_LOW'])
rule2 = ctrl.Rule(vibration['HIGH'], gps_trust['VERY_LOW'])

rule3 = ctrl.Rule(vibration['MEDIUM'], gps_trust['LOW'])
rule4 = ctrl.Rule(satellites['OKAY'] & hdop['GOOD'], gps_trust['MEDIUM'])

rule5 = ctrl.Rule(satellites['MANY'] & hdop['GOOD'] & vibration['LOW'], gps_trust['HIGH'])
rule6 = ctrl.Rule(satellites['MANY'] & hdop['EXCELLENT'] & vibration['LOW'], gps_trust['VERY_HIGH'])

gps_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
gps_simulator = ctrl.ControlSystemSimulation(gps_ctrl_system)
