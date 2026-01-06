
import predictor

print("Testing Window Size 15...")
try:
    p15 = predictor.predict_price(15)
    print(f"Prediction (15): {p15}")
except Exception as e:
    print(f"Error (15): {e}")

print("Testing Window Size 30...")
try:
    p30 = predictor.predict_price(30)
    print(f"Prediction (30): {p30}")
except Exception as e:
    print(f"Error (30): {e}")
