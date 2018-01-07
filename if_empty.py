import pandas as pd
import datetime

now = datetime.datetime.now()
filename = "result/"
date_file = str(now.date())

prediction_prob = pd.DataFrame(columns=['ID','DATE','KEYWORD','SENTENCE','POTENTIAL_VALUE'])
prediction_entity_sources = pd.DataFrame(columns=['PREDICTION_PROB_ID','SOURCE_ID','URL'])
prediction_entity = pd.DataFrame(columns=['PREDICTION_PROB_ID','NORM','ENTITY'])

prediction_prob.to_csv(filename+date_file+"_prediction_prob.csv", index = False)
prediction_entity_sources.to_csv(filename+date_file+"_prediction_entity_sources.csv", index = False)
prediction_entity.to_csv(filename+date_file+"_prediction_entity.csv", index = False)
