import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


#define max mentees/Users/ijac/side-projects/lenny_mentorship/Mentee_for_matching.csv /Users/ijac/side-projects/lenny_mentorship/Mentor_for_matching.csv
max_mentees_per_mentor = 2

#load data
mentors = pd.read_csv('Mentor_for_matching.csv')
mentees = pd.read_csv('Mentee_for_matching.csv')


#Clean DF
#define function to merge columns with same names together
def same_merge(x): return ','.join(x[x.notnull()].astype(str))

#define new DataFrame that merges columns with same names together
mentees = mentees.groupby(level=0, axis=1).apply(lambda x: x.apply(same_merge, axis=1))

print("mentor columns:", mentors.columns.values)
print("mentee columns:", mentees.columns.values)

mentors_filtered = mentors.filter(items=["Email",
                 "Offset",
                 'In-Person Meeting Location',
                 "Avg Year of YOE",
                 'Roles',
                 'Industry',
                 'Company Stage',
                 'Topics',
                 'Most Important Attribute',
                 'Created on'
                ])

mentees_filtered = mentees.filter(items=["Email",
                 "Offset",
                 'In-Person Meeting Location',
                 "Avg Year of YOE",
                 'Roles',
                 'Industry',
                 'Company Stage',
                 'Topics',
                 'Most Important Attribute',
                 'Created on'
                ])

print("mentor filter columns:", mentors_filtered.columns.values)
print("mentee filter columns:", mentees_filtered.columns.values)

#Input comma separated list of value
#Output list of values with whitespace stripped off
def clean_multiselect(x):
    if isinstance(x, str):
        return list(map(str.strip, x.split(',')))
    else:
        return []

#Input DataFrame and multi-select field to Binarize
def MultiLabelBinarize_df(input_frame, column_name):
    nested_list = list(map(clean_multiselect, input_frame[column_name].to_list()))
    mlb = MultiLabelBinarizer()
    mlb_df = pd.DataFrame(mlb.fit_transform(nested_list), columns=mlb.classes_)
    bigger = pd.concat([input_frame, mlb_df], axis=1)
    return bigger
  
  
class MultiSelect:
    def __init__(self, data=['empty']):
        if isinstance(data, str):
            self.data = clean_multiselect(data)
        else:
            self.data = data

    def __repr__(self):
        return repr(self.data)

class DistanceEstimator:
    def __init__(self, mentor_mentee_question_mapping=[]):
        self.mentor_mentee_question_mapping = mentor_mentee_question_mapping
        
    def multiSelectDistance(self, row, mentee_selection, mentor_selection):
        distance_score = 0
        matched = []
        if isinstance(mentee_selection, list) and isinstance(mentor_selection, list):
            for selection in mentee_selection:
                if selection in mentor_selection:
                    distance_score -= 10
                    matched.append(selection)
        return distance_score, matched
    
    def yoeDistance(self, mentor_yoe, mentee_yoe):
        difference = mentor_yoe - mentee_yoe
        if difference >= 8:
            return 50
        elif 4 <= difference < 8:
            return 100
        elif 2 <= difference < 4:
            return 160
        else:  # difference <= 1 or mentor_yoe <= mentee_yoe
            return -1000
    
    def _estimateDistance(self, row):
        matched = []
        distance_score = 1000
        
        for mapping in self.mentor_mentee_question_mapping:
            if mapping['mentee_question'] == mapping['mentor_question']:
                mentee_question = mapping['mentee_question'] + "-mentee"
                mentor_question = mapping['mentee_question'] + '-mentor'
            else:
                mentee_question = mapping['mentee_question']
                mentor_question = mapping['mentor_question']
                
            if mapping['question_type'] == 'multi-select':
                mentee_selection = row[mentee_question].data
                mentor_selection = row[mentor_question].data

                distance_score_temp, matched_temp = self.multiSelectDistance(row, mentee_selection, mentor_selection)

                distance_score += distance_score_temp * mapping['question_weight']
                matched += matched_temp
        
        mentor_yoe = float(row["Avg Year of YOE-mentor"])
        mentee_yoe = float(row["Avg Year of YOE-mentee"])
        distance_score -= self.yoeDistance(mentor_yoe, mentee_yoe)
    
        return distance_score, MultiSelect(matched)

    def estimateDistance(self, row):
        distance_score, matched = self._estimateDistance(row)
        return distance_score

    def matched(self, row):
        distance_score, matched = self._estimateDistance(row)
        return matched

print("mentor column value:", mentors_filtered.columns.values)
print("mentee column value:", mentees_filtered.columns.values)

mentor_mentee_question_mapping = [{'mentee_question': 'Offset',
                                   'mentor_question': 'Offset',
                                   'question_type': 'multi-select',
                                   'question_weight': 2,},
                                  {'mentee_question': 'In-Person Meeting Location',
                                   'mentor_question': 'In-Person Meeting Location',
                                   'question_type': 'multi-select',
                                   'question_weight': 1,},
                                  {'mentee_question': 'Roles',
                                   'mentor_question': 'Roles',
                                   'question_type': 'multi-select',
                                   'question_weight': 8,},
                                  {'mentee_question': 'Industry',
                                   'mentor_question': 'Industry',
                                   'question_type': 'multi-select',
                                   'question_weight': 6,},
                                  {'mentee_question': 'Company Stage',
                                   'mentor_question': 'Company Stage',
                                   'question_type': 'multi-select',
                                   'question_weight': 5,},
                                  {'mentee_question': 'Topics',
                                   'mentor_question': 'Topics',
                                   'question_type': 'multi-select',
                                   'question_weight': 7,}
                                  ]

for mapping in mentor_mentee_question_mapping:
    if mapping['question_type'] == 'multi-select':
        mentees_filtered[mapping['mentee_question']] = mentees_filtered[mapping['mentee_question']].apply(MultiSelect)
        mentors_filtered[mapping['mentor_question']] = mentors_filtered[mapping['mentor_question']].apply(MultiSelect)


combined = mentors_filtered.join(mentees_filtered, how='cross', lsuffix='-mentor', rsuffix='-mentee')

#Distance Estimation
dE = DistanceEstimator(mentor_mentee_question_mapping)
combined['distance_score'] = combined.apply(dE.estimateDistance, axis='columns')
combined['matched_criteria'] = combined.apply(dE.matched, axis='columns')
combined = combined.sort_values(by=['distance_score'])

# Matching Process:
def match_pairs(matched_list, combined, max_mentees_per_mentor):
    mentor_id = 'Email-mentor'
    mentee_id = 'Email-mentee'
    
    matched_mentors = {}
    matched_mentees = {}

    # Initialize matched_mentors dictionary
    for mentor_email in combined[mentor_id].unique():
        matched_mentors[mentor_email] = 0

    # Function to define the conditions for a valid match
    def match_condition(row):
        mentor_yoe = float(row["Avg Year of YOE-mentor"])
        mentee_yoe = float(row["Avg Year of YOE-mentee"])
        return (
            mentor_yoe > mentee_yoe and
            row[mentor_id] not in matched_mentors and
            row[mentee_id] not in matched_mentees and
            matched_mentors[row[mentor_id]] < max_mentees_per_mentor and
            matched_mentees[row[mentee_id]] < 2 and
            row[mentee_id] != row[mentor_id]
        )
    
    # Function to update the matched pairs and append to the list
    def update_matched(row):
        matched_mentors[row[mentor_id]] += 1
        matched_mentees[row[mentee_id]] += 1
        matched_list.append({
            mentor_id: row[mentor_id],
            mentee_id: row[mentee_id],
            'distance_score': row['distance_score'],
            'matched': str(row['matched_criteria'])
        })
        return matched_list

    # Iterative approach for matching pairs
    for _, row in filter(lambda r: match_condition(r[1]), combined.iterrows()):
        update_matched(row)

    return matched_list

# Call the match_pairs function to get the updated matched list
matched_list_result = match_pairs([], combined, max_mentees_per_mentor)
results = pd.DataFrame(matched_list_result)

# Save the results to CSV files
results.to_csv('matched_list.csv', index=False)
results_wide = results.join(mentors_filtered.set_index('Email'), on=mentor_id, rsuffix='-mentor').join(mentees_filtered.set_index('Email'), on=mentee_id, lsuffix='-mentor', rsuffix='-mentee')
results_wide.to_csv('matched.csv', index=False)
results_wide.to_csv('matched_wide.csv', index=False)
