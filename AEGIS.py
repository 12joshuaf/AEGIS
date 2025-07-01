# aegis_simulation.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# ----- Neural Network Definition -----
class AegisNN(nn.Module):
    def __init__(self):
        super(AegisNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # Output: [fire_sm6, fire_ciws, deploy_chaff]
        )

    def forward(self, x):
        return self.fc(x)


# ----- Scenario Generator -----
def generate_random_scenario():
    distance = random.uniform(1, 100)         # miles
    speed = random.uniform(300, 1500)         # mph
    angle = random.uniform(0, 360)            # degrees
    target_type = random.choice([0, 1])       # 0 = aircraft, 1 = missile
    sm6_remaining = random.randint(0, 4)
    ciws_cooldown = random.choice([0.0, random.uniform(1, 10)])  # 0 = ready
    chaff_remaining = random.randint(0, 5)
    time_to_impact = distance / speed * 3600  # in seconds

    return torch.tensor([
        distance, speed, angle, target_type,
        sm6_remaining, ciws_cooldown, chaff_remaining, time_to_impact
    ], dtype=torch.float32)


# ----- Reward Function -----
def simulate_defense(action, scenario):
    fire_sm6, fire_ciws, deploy_chaff = action
    dist, speed, angle, t_type, sm6, ciws_cd, chaff, tti = scenario.tolist()

    success = False
    if fire_sm6 > 0.5 and sm6 > 0 and tti > 10:
        success = True
    elif fire_ciws > 0.5 and ciws_cd == 0 and tti < 5:
        success = True
    elif deploy_chaff > 0.5 and chaff > 0 and t_type == 1 and tti < 7:
        success = random.random() < 0.4
    else:
        success = False

    return 1.0 if success else -1.0


# ----- Training Function -----
def train_aegis(epochs=10000, save_path="aegis_model.pt"):
    model = AegisNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        scenario = generate_random_scenario()
        decision = model(scenario)

        reward = simulate_defense(decision, scenario)
        target = torch.tensor([reward, reward, reward], dtype=torch.float32)

        loss = loss_fn(decision, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


# ----- Use Trained Model -----
def use_trained_model(scenario, model_path="aegis_model.pt"):
    model = AegisNN()

    # Use weights_only=True if supported
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        decision = model(scenario)
        sm6 = decision[0].item() > 0.5
        ciws = decision[1].item() > 0.5
        chaff = decision[2].item() > 0.5

        print("Scenario:")
        print(f"  Distance:     {scenario[0]:.2f} mi")
        print(f"  Speed:        {scenario[1]:.2f} mph")
        print(f"  Angle:        {scenario[2]:.2f}°")
        print(f"  Target Type:  {'Missile' if scenario[3] == 1 else 'Aircraft'}")
        print(f"  SM-6 Left:    {scenario[4]}")
        print(f"  CIWS CD:      {scenario[5]:.2f} sec")
        print(f"  Chaff Left:   {scenario[6]}")
        print(f"  Time to Hit:  {scenario[7]:.2f} sec\n")

        print("AI Decision:")
        print(f"  Fire SM-6:     {sm6}")
        print(f"  Fire CIWS:     {ciws}")
        print(f"  Deploy Chaff:  {chaff}")

        return decision


# ----- Main -----
if __name__ == "__main__":
    print("=== Training AEGIS AI Neural Network ===")
    trained_model = train_aegis()

    print("\n=== Testing on a New Random Scenario ===")
    test_scenario = generate_random_scenario()
    use_trained_model(test_scenario)
