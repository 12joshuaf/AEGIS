import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# ----- Neural Network Definition -----
class AegisNN(nn.Module):
    def __init__(self, embed_dim=8):
        super(AegisNN, self).__init__()
        self.embeds = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(10)])

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, 10)
        embedded_features = [self.embeds[i](x[:, i:i+1]) for i in range(10)]
        x_cat = torch.cat(embedded_features, dim=1)
        return self.fc(x_cat)


# ----- Scenario Generator for multiple targets -----
def generate_random_scenario(num_targets=5):
    threats = []
    for _ in range(num_targets):
        target_type = random.choice([0, 1])  # 0 = aircraft, 1 = missile

        if target_type == 0:  # Aircraft
            distance = random.uniform(20, 100)
            speed = random.uniform(300, 700)
        else:  # Missile
            distance = random.uniform(1, 50)
            speed = random.uniform(800, 1500)

        angle = random.uniform(0, 360)
        rcs = random.uniform(0.1, 1.0)  # radar cross-section

        weather_options = ['clear', 'rain', 'fog', 'storm']
        weather = random.choice(weather_options)
        weather_map = {'clear': 0.0, 'rain': 0.33, 'fog': 0.66, 'storm': 1.0}
        weather_encoded = weather_map[weather]

        sm6_remaining = random.randint(0, 4)
        ciws_cooldown = 0.0 if random.random() < 0.6 else random.uniform(1, 10)
        chaff_remaining = random.randint(0, 5)
        time_to_impact = distance / speed * 3600  # seconds

        threat = [
            distance, speed, angle, target_type,
            sm6_remaining, ciws_cooldown, chaff_remaining,
            time_to_impact, rcs, weather_encoded
        ]
        threats.append(threat)

    return torch.tensor(threats, dtype=torch.float32)  # shape: (num_targets, 10)


# ----- Reward Function for one threat -----
def simulate_defense(action, scenario):
    fire_sm6, fire_ciws, deploy_chaff = action
    (
        dist, speed, angle, t_type, sm6, ciws_cd, chaff,
        tti, rcs, weather
    ) = scenario.tolist()

    success = False
    ciws_accuracy_penalty = 0.2 * weather
    chaff_effectiveness_boost = 0.2 * weather

    if (
        fire_sm6 > 0.5
        and sm6 > 0
        and tti > 10
        and rcs > 0.3
        and weather <= 0.66  # avoid storms
    ):
        success = True
    elif (
        fire_ciws > 0.5
        and ciws_cd == 0
        and tti < 5
        and random.random() > ciws_accuracy_penalty
    ):
        success = True
    elif (
        deploy_chaff > 0.5
        and chaff > 0
        and t_type == 1
        and tti < 7
    ):
        success = random.random() < (0.4 + chaff_effectiveness_boost)

    return 1.0 if success else -1.0


# ----- Batch reward for multiple threats -----
def simulate_defense_batch(actions, scenarios):
    total_reward = 0.0
    for action, scenario in zip(actions, scenarios):
        total_reward += simulate_defense(action, scenario)
    return total_reward / len(actions)  # average reward


# ----- Training Function -----
def train_aegis(epochs=10000, save_path="aegis_model.pt", patience=100, min_delta=0.001):
    model = AegisNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    loss_history = []
    best_avg_loss = float('inf')
    epochs_since_improvement = 0
    window = 100

    for epoch in range(epochs):
        num_targets = random.randint(1, 100)
        scenario = generate_random_scenario(num_targets)  # (N, 10)
        decisions = model(scenario)  # (N, 3)

        reward = simulate_defense_batch(decisions, scenario)

        targets = decisions.clone().detach()
        for i in range(decisions.shape[0]):
            for j in range(3):
                if decisions[i, j] > 0.5:
                    targets[i, j] = reward
                else:
                    targets[i, j] = decisions[i, j]

        loss = loss_fn(decisions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if len(loss_history) >= window:
            avg_loss = sum(loss_history[-window:]) / window
            delta = best_avg_loss - avg_loss

            if delta < min_delta:
                epochs_since_improvement += 1
            else:
                best_avg_loss = avg_loss
                epochs_since_improvement = 0

            if epochs_since_improvement >= patience:
                print(f"\n🛑 Training stopped early at epoch {epoch} due to learning saturation.")
                print(f"    Avg Loss: {avg_loss:.5f} | ΔLoss: {delta:.5f} | Patience: {patience}")
                break

        if epoch % 500 == 0:
            print(f"Epoch {epoch:>5} | Loss: {loss.item():.4f} | Avg (last {window}): {avg_loss if len(loss_history) >= window else '...'}")

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")

    # Plot loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    if len(loss_history) >= window:
        avg_losses = [sum(loss_history[i-window:i]) / window for i in range(window, len(loss_history)+1)]
        plt.plot(range(window-1, len(loss_history)), avg_losses, label=f"{window}-Epoch Avg", color="orange")

    plt.title("AEGIS AI Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


# ----- Use Trained Model -----
def use_trained_model(scenario, model_path="aegis_model.pt"):
    model = AegisNN()
    try:
        state_dict = torch.load(model_path, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        decisions = model(scenario)  # (N, 3)
        for i, (s, d) in enumerate(zip(scenario, decisions)):
            sm6 = d[0].item() > 0.5
            ciws = d[1].item() > 0.5
            chaff = d[2].item() > 0.5

            print(f"\nThreat #{i + 1}")
            print(f"  Distance:     {s[0]:.2f} mi")
            print(f"  Speed:        {s[1]:.2f} mph")
            print(f"  Angle:        {s[2]:.2f}°")
            print(f"  Target Type:  {'Missile' if s[3] == 1 else 'Aircraft'}")
            print(f"  SM-6 Left:    {s[4]}")
            print(f"  CIWS CD:      {s[5]:.2f} sec")
            print(f"  Chaff Left:   {s[6]}")
            print(f"  Time to Hit:  {s[7]:.2f} sec")
            print(f"  RCS:          {s[8]:.2f}")
            print(f"  Weather Code: {s[9]:.2f} (0=clear, 1=storm)")
            print("  AI Decision:")
            print(f"    Fire SM-6:     {sm6}")
            print(f"    Fire CIWS:     {ciws}")
            print(f"    Deploy Chaff:  {chaff}")

    return decisions


# ----- Main -----
if __name__ == "__main__":
    print("=== Training AEGIS AI Neural Network ===")
    trained_model = train_aegis()

    print("\n=== Testing on a New Random Scenario ===")
    test_scenario = generate_random_scenario(random.randint(1, 10))
    use_trained_model(test_scenario)
