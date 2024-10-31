import pandas as pd

def load_gait_data(file_path):
    df = pd.read_csv(file_path)
    # Assume columns ['timestamp', 'acc_x', 'acc_y', 'acc_z']
    gait_features = df[['acc_x', 'acc_y', 'acc_z']].values
    return gait_features

# Example usage:
gait_features_list = []

gait_data_dir = 'path_to_gait_data'  # Replace with your path

for filename in os.listdir(gait_data_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(gait_data_dir, filename)
        features = load_gait_data(file_path)
        gait_features_list.append(features)