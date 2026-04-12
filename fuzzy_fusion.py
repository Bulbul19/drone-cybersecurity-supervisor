class FuzzyTrustSystem:
    def __init__(self):
        pass

    def _trapezoid(self, x, a, b, c, d):
        """Standard Fuzzy Trapezoid shape."""
        if x <= a or x >= d: return 0.0
        if x >= b and x <= c: return 1.0
        if x < b: return (x - a) / (b - a)
        return (d - x) / (d - c)

    def evaluate_imu_trust(self, vibration, gyro_mag):
        """
        Logic:
        - If Vibration is low (<0.1g) AND Gyro is steady -> HIGH TRUST
        - If Vibration is High (>0.5g) OR Gyro spinning -> LOW TRUST
        """
        # 1. Vibration Trust (Low is good)
        # 0.0 to 0.1g = Perfect, 0.5g = Bad
        vib_score = self._trapezoid(vibration, -1.0, -0.5, 0.1, 0.5) 
        
        # 2. Gyro Trust (Low rotation is usually more reliable for position holding)
        # 0 deg/s = Perfect, 100 deg/s = Chaotic
        gyro_score = self._trapezoid(gyro_mag, -100, -50, 20, 100)
        
        # Combine (Conservative: take the lowest score)
        # We invert the logic because trapezoid was set for "High" value
        # Let's simplify:
        
        # RE-WRITE for clarity:
        # Trust is 1.0 if vibration is 0. Trust is 0.0 if vibration > 0.4
        trust_v = max(0, 1.0 - (vibration * 2.5)) 
        
        # Trust is 1.0 if gyro < 10. Trust is 0.0 if gyro > 200
        trust_g = max(0, 1.0 - (gyro_mag / 200.0))
        
        return min(trust_v, trust_g)

    def evaluate_baro_trust(self, velocity_variance):
        """
        Logic:
        - Barometers cannot physically jump 10 meters in 0.1 seconds.
        - High velocity variance = Sensor noise or wind gust -> LOW TRUST
        """
        # If variance is < 2 m/s, Trust = 1.0
        # If variance is > 5 m/s, Trust = 0.0
        if velocity_variance < 2.0: return 1.0
        if velocity_variance > 5.0: return 0.0
        
        return 1.0 - ((velocity_variance - 2.0) / 3.0)

    def evaluate_gps_trust(self, sats, hdop):
        """
        Logic:
        - Sats > 12 and HDOP < 1.0 -> HIGH TRUST
        """
        if sats < 6: return 0.0
        if hdop > 3.0: return 0.0
        
        # HDOP score (Lower is better)
        hdop_trust = max(0, 1.0 - (hdop / 3.0))
        
        # Sat score (Higher is better, maxing at 20)
        sat_trust = min(1.0, sats / 16.0)
        
        return (hdop_trust * 0.6) + (sat_trust * 0.4)

    def fuse_altitude(self, baro_alt, baro_trust, gps_alt, gps_trust):
        """
        Weighted Average Fusion based on Trust Scores.
        """
        total_trust = baro_trust + gps_trust
        
        if total_trust == 0:
            return baro_alt # Fallback to Baro if both fail
            
        fused_val = (baro_alt * baro_trust) + (gps_alt * gps_trust)
        return fused_val / total_trust
