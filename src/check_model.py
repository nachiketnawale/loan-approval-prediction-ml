import joblib

model = joblib.load("loan_approval_model.pkl")

print("Model type:")
print(type(model))

print("\nPipeline steps:")

for name, step in model.named_steps.items():
    print(name, "->", type(step))