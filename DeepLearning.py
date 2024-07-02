import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

data = pd.read_excel('Rankings_and_results_v_1.xlsx')
team_list = [
            "Argentina",
            "Australia",
            "Chile",
            "England",
            "Fiji",
            "France",
            "Georgia",
            "Ireland",
            "Italy",
            "Japan",
            "Namibia",
            "New Zealand",
            "Portugal",
            "Romania",
            "Samoa",
            "Scotland",
            "South Africa",
            "Tonga",
            "Uruguay",
            "Wales"]

filter_team1 = data[data['Team1'].isin(team_list)]
filter_team2 = data[data['Team2'].isin(team_list)]

date_sorted = data.sort_values(by='Date', ascending=True)

mask = date_sorted['Date'] >= '2021-01-01'

# Apply the mask to filter out rows with dates before 2019
filtered_df = date_sorted[mask]

# # Calculate the mean and standard deviation of the score difference
# new = date_sorted.assign(ScoreDifference = date_sorted['Team1Score'] - date_sorted['Team2Score'] )
# print(f"{new=}")
# mean_score_diff = new['ScoreDifference'].mean()
# std_score_diff = new['ScoreDifference'].std()

# # Define the range as mean ± 3 * standard deviation
# lower_bound = mean_score_diff - 1 * std_score_diff
# upper_bound = mean_score_diff + 1 * std_score_diff

# print(f"{lower_bound=}")
# print(f"{upper_bound=}")

# # Filter the DataFrame to include only data points within the specified range
# filtered_df = new[(new['ScoreDifference'] >= lower_bound) & (new['ScoreDifference'] <= upper_bound)]
 
# # Drop the temporary 'ScoreDifference' column
# filtered_df = new.drop(columns=['ScoreDifference'])

# print(f"{filtered_df=}")

U = filtered_df.drop(['Team1Score', 'Team2Score','Date'], axis=1)  # Features excluding scores

Y_score = filtered_df[['Team1Score', 'Team2Score']]  # Target variables
print(Y_score)

# output_scaler = MinMaxScaler()
# Y_score_scaled = output_scaler.fit_transform(Y_score)

# print(f"{Y_score_scaled=}")

encoder = ce.BinaryEncoder(cols=['Team1', 'Team2'], verbose = 1)
U_encoded = encoder.fit_transform(U)

# U_encoded = pd.get_dummies(U, columns=['Team1', 'Team2'], drop_first=True,dtype=float)

# Initialize the Min-Max scaler
scaler = MinMaxScaler()

# # Fit and transform the numerical features
# U_encoded[['Team1Pos', 'Team1Pts', 'Team2Pos', 'Team2Pts']] = scaler.fit_transform(
#     U_encoded[['Team1Pos', 'Team1Pts', 'Team2Pos', 'Team2Pts']]
# )

print(f"{U_encoded=}")

# # Convert the NumPy array to a DataFrame
# df = pd.DataFrame(U_encoded)

# # Specify the file path where you want to save the CSV file
# file_path = 'encoded.csv'

# # Create a KFold object
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Create a sequential model
model = tf.keras.Sequential()
 
# Add input layer (adjust input_shape based on the number of input features)
model.add(tf.keras.layers.Dense(units=15,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=12)
, activation='relu', input_shape=(U_encoded.shape[1],)))

model.add(tf.keras.layers.Dense(units=8,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=14)
, activation='relu', input_shape=(U_encoded.shape[1],)))

# model.add(tf.keras.layers.Dense(units=14,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=18)
# , activation='relu', input_shape=(U_encoded.shape[1],)))

# Add output layer for score prediction (2 neurons)
model.add(tf.keras.layers.Dense(units=2, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=255)
, name='score_prediction'))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=100,
    decay_rate=0.6)
optimizer_1 = tf.keras.optimizers.Adam( beta_1=0.9,
                                        beta_2=0.9,
                                        epsilon=1e-02, 
                                        learning_rate=lr_schedule)


optimizer2= tf.keras.optimizers.Adamax(learning_rate=lr_schedule,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-02)


# Compile the model with the appropriate loss function for score prediction
model.compile(loss="mean_squared_error", optimizer=optimizer2)

# Initialize lists to store training and validation errors for each fold
train_losses = []
validation_losses = []

num_epochs = 90
batch_size = 22

index = int(0.95*U_encoded.shape[0])
print(index)
U_train, U_test = U_encoded[0:index], U_encoded[index+1:U_encoded.shape[0]]
y_score_train, y_score_test = Y_score[0:index], Y_score[index+1:U_encoded.shape[0]]

# Train the model
history = model.fit(U_train, y_score_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=0.05)

results = model.evaluate(U_test, y_score_test, batch_size=128)
print("test loss, test acc:", results)



predictions = model.predict(U_test)

#rescaled = output_scaler.inverse_transform(predictions)
for i in range(len(y_score_test)):
    print("----------------------------------------------------------------------------------------------")
    print(predictions[i]) 
    diff = predictions[i][0] -predictions[i][1]
    print('['+ str(y_score_test.iloc[i]["Team1Score"]) + ' ' + str(y_score_test.iloc[i]["Team2Score"]) + ']')
    act = y_score_test.iloc[i]["Team1Score"] - y_score_test.iloc[i]["Team2Score"]
    print('predicted = ' + str(diff) + ' actual =' + str(act) )

# # Train a RandomForestRegressor
# rf_model = RandomForestRegressor()
# rf_model.fit(U_train, y_score_train)

# # Get feature importances
# feature_importances = rf_model.feature_importances_

# # Pair feature names with their importance scores
# feature_names = list(U_train.columns)  # Replace with your actual feature names
# feature_importance_dict = dict(zip(feature_names, feature_importances))

# # Sort features by importance (descending order)
# sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# # Print or visualize the sorted feature importance
# for feature, importance in sorted_feature_importance:
#     print(f"{feature}: {importance}")

# # # Calculate the average training and validation losses across all folds
# # average_train_loss = np.mean(train_losses, axis=0)
# # average_validation_loss = np.mean(validation_losses, axis=0)

# # # Plot training and validation losses
# # plt.figure(figsize=(8, 6))
# # epochs = range(1, len(average_train_loss) + 1)
# # plt.plot(epochs, average_train_loss, marker='o', linestyle='-', color='b', label='Training Loss')
# # plt.plot(epochs, average_validation_loss, marker='o', linestyle='-', color='r', label='Validation Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Mean Squared Error (MSE)')
# # plt.title('Training and Validation Loss Across Folds')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
