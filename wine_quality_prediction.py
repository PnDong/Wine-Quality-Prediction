# wine_quality_prediction.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def plot_all_columns(df):
    # 3x4 subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    # initialize i and j
    i = j = 0
    # plot for all columns
    for c in df.iloc[:, :-1]:
        axes[i, j].scatter(df[c], df['quality'])
        axes[i, j].set_title(c)
        j += 1
        if j > 3:
            j = 0
            i += 1
    plt.show()


def corr_rp(matrix):
    # get shape
    rows, cols = matrix.shape[0], matrix.shape[1]
    # initialize matrices with 1
    r = np.ones(shape=(cols, cols))
    p = np.zeros(shape=(cols, cols))
    # get pearson correlation matrix and the p-value
    for i in range(cols):
        for j in range(i + 1, cols):
            r_, p_ = pearsonr(matrix[:, i], matrix[:, j])
            r[i, j] = r[j, i] = r_
            p[i, j] = p[j, i] = p_
    return r, p


# read wine data
red_wine_df = pd.read_csv('winequality-red.csv', sep=';')
white_wine_df = pd.read_csv('winequality-white.csv', sep=';')

'''
White wine
'''

# Make a train/test split using 30% test size
white_wine_train, white_wine_test = train_test_split(white_wine_df, test_size=0.30, random_state=7)

# plot the target var wrt each predictor vars
# plot_all_columns(white_wine_df)

# get pearson correlation matrix and the two-tailed p-value for hypothesis testing
# r is the correlation matrix and p is the p-value matrix
# print out 12th column of each matrix, which is about target variable
# you may use the entire matrix, i.e. r and p, for plotting
r, p = corr_rp(white_wine_df.to_numpy())
print("Corr of 12th column (White wine):\n", r[:, 11])
print("Pval of 12th column (White wine):\n", p[:, 11])

# scale predictors by its standard deviation
white_wine_train_pred = white_wine_train.iloc[:, :-1]
white_wine_train_target = white_wine_train.iloc[:, -1]
std_scaler = StandardScaler()
fitted = std_scaler.fit(white_wine_train_pred)
scaled_pred = pd.DataFrame(std_scaler.transform(white_wine_train_pred),
                           columns=white_wine_train_pred.columns,
                           index=list(white_wine_train_pred.index.values))

# PCA projection to 2D
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(scaled_pred)
pca_transformed_df = pd.DataFrame(data=pca_transformed, columns=['principal component 1', 'principal component 2'],
                                  index=list(scaled_pred.index.values))
final_df = pd.concat([pca_transformed_df, white_wine_train_target], axis=1)

# display PCA projection to 2D
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
# adjust limit of x-axis and y-axis as you want
# ax.set_xlim([-150, 150])
# ax.set_ylim([-50, 100])
ax.set_title('2 Component PCA (White wine)', fontsize=15)

for qual in range(1, 11):
    logical_inx = final_df['quality'] == qual
    # set the color as you want
    color = ""
    if qual <= 4:
        color = "#FF0000"
    elif qual == 6:
        color = ""
    elif qual >= 7:
        color = "#0000FF"
    sc = ax.scatter(final_df.loc[logical_inx, 'principal component 1'],
                    final_df.loc[logical_inx, 'principal component 2'],
                    edgecolors=color,
                    facecolors='none',
                    s=35)
ax.grid()
plt.show()

# PCA projection to 11D (full principal components)
pca_full = PCA(n_components=11)
pca_transformed = pca_full.fit_transform(scaled_pred)
pca_transformed_df = pd.DataFrame(data=pca_transformed, columns=['PC 1', 'PC 2',
                                                                 'PC 3', 'PC 4',
                                                                 'PC 5', 'PC 6',
                                                                 'PC 7', 'PC 8',
                                                                 'PC 9', 'PC 10',
                                                                 'PC 11'],
                                  index=list(scaled_pred.index.values))

# print out POV of all principal components
print('Proportion of Variance Explained (White wine):\n', pca_full.explained_variance_ratio_)

# visualize PoV of all principal components
fig = plt.figure(figsize=(7, 5))
percentage_of_variance = pca_full.explained_variance_ratio_

plt.plot(pca_transformed_df.columns, percentage_of_variance, 'ro-', linewidth=2, )
plt.title('Proportion of Variance Explained (White wine)')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# print out all principal components and its weights
print(pd.DataFrame(pca_full.components_, columns=white_wine_train_pred.columns,
                   index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10', 'PC-11']))

# scale predictors by its standard deviation
white_wine_test_pred = white_wine_test.iloc[:, :-1]
white_wine_test_target = white_wine_test.iloc[:, -1]
std_scaler_test = StandardScaler()
fitted_test = std_scaler.fit(white_wine_test_pred)
scaled_pred_test = pd.DataFrame(std_scaler.transform(white_wine_test_pred),
                                columns=white_wine_test_pred.columns,
                                index=list(white_wine_test_pred.index.values))

# prediction models
prediction_models = [
    LinearRegression(),
    MLPRegressor(hidden_layer_sizes=[8], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[16], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[16, 4], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[32, 16, 4], max_iter=2000, alpha=0.005, random_state=7)
]

# all features
i = 0
print("\nModels using all features\n")
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(scaled_pred, white_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(scaled_pred) - white_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(scaled_pred_test) - white_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f" % np.mean((pm.predict(scaled_pred) - white_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(scaled_pred, white_wine_train_target))
    print("MSE (Generalization): %.6f" % np.mean((pm.predict(scaled_pred_test) - white_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(scaled_pred_test, white_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()

# selected features
selected_features = ['alcohol', 'volatile acidity', 'density', 'total sulfur dioxide', 'chlorides']
i = 0
print("\nModels using selected features : \n", selected_features)
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(scaled_pred[selected_features], white_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(scaled_pred[selected_features]) - white_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(scaled_pred_test[selected_features]) - white_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f"
          % np.mean((pm.predict(scaled_pred[selected_features]) - white_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(scaled_pred[selected_features], white_wine_train_target))
    print("MSE (Generalization): %.6f"
          % np.mean((pm.predict(scaled_pred_test[selected_features]) - white_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(scaled_pred_test[selected_features], white_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()

# PCA projection to 4D
pca = PCA(n_components=4)
pca_transformed = pca.fit_transform(scaled_pred)

# models using PCA
i = 0
print("\nModels using 4 Principal Components\n")
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(pca_transformed, white_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(pca_transformed) - white_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(pca.transform(scaled_pred_test)) - white_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f" % np.mean((pm.predict(pca_transformed) - white_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(pca_transformed, white_wine_train_target))
    print("MSE (Generalization): %.6f"
          % np.mean((pm.predict(pca.transform(scaled_pred_test)) - white_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(pca.transform(scaled_pred_test), white_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()

'''
Red wine
'''

# Make a train/test split using 30% test siz
red_wine_train, red_wine_test = train_test_split(red_wine_df, test_size=0.30, random_state=7)

# plot the target var wrt each predictor vars
# plot_all_columns(red_wine_df)

# get pearson correlation matrix and the two-tailed p-value for hypothesis testing
# r is the correlation matrix and p is the p-value matrix
# print out 12th column of each matrix, which is about target variable
# you may use the entire matrix, i.e. r and p, for plotting
r, p = corr_rp(red_wine_df.to_numpy())
print("corr of 12th column (Red wine):\n", r[:, 11])
print("pval of 12th column (Red wine):\n", p[:, 11])

# scale predictors by its standard deviation
red_wine_train_pred = red_wine_train.iloc[:, :-1]
red_wine_train_target = red_wine_train.iloc[:, -1]
std_scaler = StandardScaler()
fitted = std_scaler.fit(red_wine_train_pred)
scaled_pred = pd.DataFrame(std_scaler.transform(red_wine_train_pred),
                           columns=red_wine_train_pred.columns,
                           index=list(red_wine_train_pred.index.values))

# PCA projection to 2D
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(scaled_pred)
pca_transformed_df = pd.DataFrame(data=pca_transformed, columns=['principal component 1', 'principal component 2'],
                                  index=list(scaled_pred.index.values))
final_df = pd.concat([pca_transformed_df, red_wine_train_target], axis=1)

# display PCA projection to 2D
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=12)
ax.set_ylabel('Principal Component 2', fontsize=12)
# adjust limit of x-axis and y-axis as you want
# ax.set_xlim([-150, 150])
# ax.set_ylim([-50, 100])
ax.set_title('2 Component PCA (Red wine)', fontsize=15)

for qual in range(1, 11):
    logical_inx = final_df['quality'] == qual
    # set the color as you want
    color = ""
    if qual <= 4:
        color = "#FF0000"
    elif qual == 6:
        color = ""
    elif qual >= 7:
        color = "#0000FF"
    sc = ax.scatter(final_df.loc[logical_inx, 'principal component 1'],
                    final_df.loc[logical_inx, 'principal component 2'],
                    edgecolors=color,
                    facecolors='none',
                    s=35)
ax.grid()
plt.show()

# PCA projection to 11D (full principal components)
pca_full = PCA(n_components=11)
pca_transformed = pca_full.fit_transform(scaled_pred)
pca_transformed_df = pd.DataFrame(data=pca_transformed, columns=['PC 1', 'PC 2',
                                                                 'PC 3', 'PC 4',
                                                                 'PC 5', 'PC 6',
                                                                 'PC 7', 'PC 8',
                                                                 'PC 9', 'PC 10',
                                                                 'PC 11'],
                                  index=list(scaled_pred.index.values))

# print out POV of all principal components
print('Proportion of Variance Explained (Red wine):\n', pca_full.explained_variance_ratio_)

# visualize PoV of all principal components
fig = plt.figure(figsize=(7, 5))
percentage_of_variance = pca_full.explained_variance_ratio_

plt.plot(pca_transformed_df.columns, percentage_of_variance, 'ro-', linewidth=2, )
plt.title('Proportion of Variance Explained (Red wine)')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

# print out all principal components and its weights
print(pd.DataFrame(pca_full.components_, columns=red_wine_train_pred.columns,
                   index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7', 'PC-8', 'PC-9', 'PC-10', 'PC-11']))

# scale predictors by its standard deviation
red_wine_test_pred = red_wine_test.iloc[:, :-1]
red_wine_test_target = red_wine_test.iloc[:, -1]
std_scaler_test = StandardScaler()
fitted_test = std_scaler.fit(red_wine_test_pred)
scaled_pred_test = pd.DataFrame(std_scaler.transform(red_wine_test_pred),
                                columns=red_wine_test_pred.columns,
                                index=list(red_wine_test_pred.index.values))

# prediction models
prediction_models = [
    LinearRegression(),
    MLPRegressor(hidden_layer_sizes=[8], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[16], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[16, 4], max_iter=2000, alpha=0.005, random_state=7),
    MLPRegressor(hidden_layer_sizes=[32, 16, 4], max_iter=2000, alpha=0.005, random_state=7)
]

# all features
i = 0
print("\nModels using all features\n")
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(scaled_pred, red_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(scaled_pred) - red_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(scaled_pred_test) - red_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f" % np.mean((pm.predict(scaled_pred) - red_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(scaled_pred, red_wine_train_target))
    print("MSE (Generalization): %.6f" % np.mean((pm.predict(scaled_pred_test) - red_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(scaled_pred_test, red_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()

# selected features
selected_features = ['alcohol', 'volatile acidity']
i = 0
print("\nModels using selected features : \n", selected_features)
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(scaled_pred[selected_features], red_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(scaled_pred[selected_features]) - red_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(scaled_pred_test[selected_features]) - red_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f" % np.mean((pm.predict(scaled_pred[selected_features]) - red_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(scaled_pred[selected_features], red_wine_train_target))
    print("MSE (Generalization): %.6f"
          % np.mean((pm.predict(scaled_pred_test[selected_features]) - red_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(scaled_pred_test[selected_features], red_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()

# PCA projection to 2D
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(scaled_pred)

# models using PCA
i = 0
print("\nModels using 2 Principal Components\n")
all_mse = np.zeros(shape=(2, 5))
for pm in prediction_models:
    pm.fit(pca_transformed, red_wine_train_target)
    all_mse[0, i] = np.mean((pm.predict(pca_transformed) - red_wine_train_target) ** 2)
    all_mse[1, i] = np.mean((pm.predict(pca.transform(scaled_pred_test)) - red_wine_test_target) ** 2)
    i = i + 1
    print("Model #", i)
    print(pm.__class__)
    print("MSE (Empirical): %.6f" % np.mean((pm.predict(pca_transformed) - red_wine_train_target) ** 2))
    print('R2 score: %.2f' % pm.score(pca_transformed, red_wine_train_target))
    print("MSE (Generalization): %.6f"
          % np.mean((pm.predict(pca.transform(scaled_pred_test)) - red_wine_test_target) ** 2))
    print('R2 score: %.2f\n' % pm.score(pca.transform(scaled_pred_test), red_wine_test_target))

# visualize MSEs
fig = plt.figure(figsize=(7, 5))
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[0, 0], 4), 'r-', linewidth=1, label='Training Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], np.repeat(all_mse[1, 0], 4), 'r--', linewidth=1, label='Validation Error of MLR')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[0, 1:], 'b.-', linewidth=1, label='Training Error of MLP')
plt.plot(['[8]', '[16]', '[16,4]', '[32,16,4]'], all_mse[1, 1:], 'b.--', linewidth=1, label='Validation Error of MLP')
plt.title('MSE of the models')
plt.xlabel('Hidden Layer')
plt.ylabel('MSE')
plt.legend()
plt.show()
