import io
import keras
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

mushroom_dataset_raw = pd.read_csv("mushroom_data.csv")

mushroom_dataset = mushroom_dataset_raw[[
    'cap_diameter',
    'stem_height',
    'spore_density',
    'color_score',
    'edible'
]]

#print(mushroom_dataset.describe())
#print(
#    f'The biggest cap diameter is {mushroom_dataset.cap_diameter.max():.1f},'
#    f' while the smallest is {mushroom_dataset.cap_diameter.min():.1f}.'
#)

# print(f"\n{mushroom_dataset.corr(numeric_only = True)}")
#
# for x_axis_data, y_axis_data in [
#     ('spore_density', 'cap_diameter')
# ]:
#   px.scatter(mushroom_dataset, x=x_axis_data, y=y_axis_data, color='edible').show()

# px.scatter_3d(
#     mushroom_dataset,
#     x='spore_density',
#     y='cap_diameter',
#     z='stem_height',
#     color='edible',
# ).show()

feature_mean = mushroom_dataset.mean(numeric_only=True)
feature_std = mushroom_dataset.std(numeric_only=True)
numerical_features = mushroom_dataset.select_dtypes('number').columns
normalized_dataset = (
    mushroom_dataset[numerical_features] - feature_mean
) / feature_std

normalized_dataset['edible'] = mushroom_dataset['edible']
#print(normalized_dataset.head())

keras.utils.set_random_seed(67)

#print(normalized_dataset.sample(10))

number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

#print(test_data.head())

label_columns = ['edible']

train_features = train_data.drop(columns=label_columns)
train_labels = train_data['edible'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['edible'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['edible'].to_numpy()

input_features = [
    'spore_density',
    'cap_diameter',
    'stem_height'
]

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.3,
    number_epochs=50,
    batch_size=100,
    classification_threshold=0.9,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model = create_model(settings, metrics)

experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
plt.show()
ml_edu.results.plot_experiment_metrics(experiment, ['auc'])
plt.show()

def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
  print('Comparing metrics between train and test:')
  for metric, test_value in test_metrics.items():
    print('------')
    print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
    print(f'Test {metric}:  {test_value:.4f}')

test_metrics = experiment.evaluate(test_features, test_labels)
compare_train_test(experiment, test_metrics)

all_input_features = [
  'cap_diameter',
  'stem_height',
  'spore_density',
  'color_score'
]

settings_all_features = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.3,
    number_epochs=50,
    batch_size=100,
    classification_threshold=0.9,
    input_features=all_input_features
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy',
        threshold=settings_all_features.classification_threshold,
    ),
    keras.metrics.Precision(
        name='precision',
        thresholds=settings_all_features.classification_threshold,
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings_all_features.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

model_all_features = create_model(settings_all_features, metrics)

experiment_all_features = train_model(
    'all features',
    model_all_features,
    train_features,
    train_labels,
    settings_all_features,
)

ml_edu.results.plot_experiment_metrics(
    experiment_all_features, ['accuracy', 'precision', 'recall']
)
plt.show()
ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])
plt.show()

test_metrics_all_features = experiment_all_features.evaluate(
    test_features,
    test_labels,
)
compare_train_test(experiment_all_features, test_metrics_all_features)

ml_edu.results.compare_experiment([experiment, experiment_all_features],
                                  ['accuracy', 'auc'],
                                  test_features, test_labels)

def predict_edibility(model, settings, raw_sample):
    input_data = {}
    for feature in settings.input_features:
        mean = feature_mean[feature]
        std = feature_std[feature]
        input_value = raw_sample[feature]
        normalized_value = (input_value - mean) / std
        input_data[feature] = np.array([[normalized_value]])

    probability = model.predict(input_data)[0][0]
    is_edible = probability >= settings.classification_threshold

    print(f"Probability of being edible: {probability:.4f}")
    return "Edible" if is_edible else "Poisonous"

sample_input = {'cap_diameter': float(input('Enter cap diameter: ')),
                'stem_height': float(input('Enter stem height: ')),
                'spore_density': float(input('Enter spore density: ')),
                'color_score': float(input('Enter color score: '))}

print('')
result = predict_edibility(model_all_features, settings_all_features, sample_input)
print("Prediction result:", result)