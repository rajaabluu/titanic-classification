import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        X = X.copy()
        
        titles = X['Name'].apply(self._extract_title)
        
        self.title_age_map_ = X.groupby(titles)['Age'].mean()
        
        return self
        
    def transform(self, X, y=None):
        X = X.copy()
        
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['Fare'] = X['Fare'].apply(np.log1p)

        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        
        X['Title'] = X['Name'].apply(self._extract_title)
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')

        X['Age'] = X['Age'].fillna(X['Title'].map(self.title_age_map_))
        X['Cabin'] = X['Cabin'].fillna('Unknown')
        X['Deck'] = X['Cabin'].str[0]
        X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        X['FarePerPerson'] = X['Fare'] / X['FamilySize']
        X['TicketGroupSize'] = X.groupby('Ticket')['Ticket'].transform('count')
        
        return X.drop(columns=['SibSp', 'Parch', 'Ticket', 'Cabin'], errors='ignore')
    
    @staticmethod
    def _extract_title(name: str):
        if pd.isna(name) or ',' not in name or '.' not in name:
            return 'Unknown'
        title = name.split(',')[1].split('.')[0]
        return title.strip()