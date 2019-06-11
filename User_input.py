########################################
####Get Data set########################
########################################
import re
import nltk as n
from nltk.stem import *
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import ISRIStemmer
import xlrd
import pickle

Dataset_Dictionary = {0: {'Question': '', 'Answer': '', 'Label': 0}}
Tokens_Dataset_Dictionary = {0: {'Question': [], 'Answer': [], 'Label': 0}}
#loc = ("CRA Dataset.csv")
for i in range(2179):
     wb = xlrd.open_workbook("CRA-Dataset.xlsx")
     sheet = wb.sheet_by_index(0)
     sheet.cell_value(i, 0)
     Dataset_Dictionary[i] = {'Question': sheet.cell_value(i, 0), 'Answer': sheet.cell_value(i, 1) , 'Label': sheet.cell_value(i, 2)}
################################
#### user input features########
################################


def user_features(preprocessed_data):
  features_names = 0
  with open('Question_features_names.tmp', 'rb') as dic:
      features_names = pickle.load(dic)
  #print(features_names)
  #print(len(Q_vectorizer.get_feature_names()))
  pr = preprocessing_test_data(preprocessed_data)
  QU = ''
  for i in range(len(pr)):
    QU += pr[i]
    QU += ' '
  #print(QU)
  QU_list = [QU]
  QU_vectorizer = TfidfVectorizer()
  QU_vectorizer.fit(QU_list)
  #print(QU_vectorizer.get_feature_names())
  #print(QU_vectorizer.idf_)
  UQuestion_features = QU_vectorizer.transform(QU_list)
  #print(UQuestion_features[0, 1])
  #print(Q_vectorizer.get_feature_names())
  QU_features = np.zeros((1, 357))
  for j in range(357):
    for k in range(len(QU_vectorizer.get_feature_names())):
      if(QU_vectorizer.get_feature_names()[k] == features_names[j]):
        QU_features[0, j] = UQuestion_features[0, k]
  print(QU_features.shape)
  return QU_features 

features = user_features("لو سمحت ما سعر وصلة سريعة ؟")
print(features.shape)
#print(question_features_matrix.shape)
# define layers
tensorflow.reset_default_graph()
net = tf.input_data(shape=[None, 357], name='input')
net = tf.fully_connected(net, 256, activation='relu')
net = tf.fully_connected(net, 128, activation='relu')
net = tf.fully_connected(net, 64, activation='relu')
output = tf.fully_connected(net, 2179, activation='softmax')
output = tf.regression(output, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='output')

#define model
model = tf.DNN(output)

if(os.path.exists('model.tfl.meta')):
  model.load('model.tfl', weights_only=True)  
  print("exists")
else:
  print("does not exist")
  model.fit({'input': x_train}, {'output': y_train}, batch_size=2179, n_epoch=500,
        snapshot_step=None, show_metric=True, run_id='Model-01')
  model.save('tf_learn_model/model.tfl')
  
#features = np.array([i for i in features]).reshape(1, 1, 357)
prediction = model.predict(features)
print(prediction)
predictions_list = []
for i in range(len(prediction)):
      predictions_list.append(np.argmax(prediction[i]) + 1)
print(predictions_list)
print(Dataset_Dictionary[predictions_list[0]]['Answer'])
