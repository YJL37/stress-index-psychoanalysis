# cleaning up the data:
#   - combining section 2 into one numerical score
#   - getting rid of column with dates and times

import pandas as pd

class preprocess():
    def __init__(self):
        self.df = pd.read_csv('data.csv')
    
    def convert_to_score(self):
        # Calculates physical symptoms into a single score
        scoring = {'Never': 0, 'Sometimes': 1, 'Often': 2, 'Very often': 3}
        scores = []

        for i in range(self.df.shape[0]):
            cnt = 0

            for j in self.df.columns:
                if self.df.get(j)[i] in scoring.keys():
                    cnt += scoring[self.df.get(j)[i]]

            if 0 <= cnt < 6:
                scores.append('Low')
            elif 6 <= cnt < 12:
                scores.append('Moderate')
            elif 12 <= cnt < 18:
                scores.append('High')
            else:
                scores.append('Very high')

        #print(scores)
        self.df = self.df.assign(Score = scores)
    
    def process_columns(self):
        # Drops timestamp column and all columns that are no longer needed after convert_to_score
        self.df = self.df.drop(['Timestamp', ' [Aches and pains]', ' [Chest pain or heart pounding]', 
                                ' [Exhaustion or trouble sleeping]', ' [Headaches, dizziness or shaking]', 
                                ' [High blood pressure]', ' [Muscle tension or jaw clenching]', 
                                ' [Stomach or digestive problems]', ' [Weakened immune system]'], axis = 1)

        print(self.df)

res = preprocess()
res.convert_to_score()
res.process_columns()