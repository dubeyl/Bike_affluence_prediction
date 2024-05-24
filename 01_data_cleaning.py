import numpy as np
import pandas as pd
import os


def data_clean_year(year):
    stations = pd.read_csv(f'../Data/original/{year}_stations.csv')

    datasets = []
    for month in range(0, 13):
        month = '{:02d}'.format(month)
        file_name = f'../Data/original/{year}-{month}_opendata.csv'
        if os.path.exists(file_name):
            dataset = pd.read_csv(file_name)
            dataset = pd.merge(dataset, stations, left_on='start_station_code', right_on='code', how='left')
            dataset = pd.merge(dataset, stations, left_on='end_station_code', right_on='code', how='left')

            columns_to_delete = ['code_x', 'code_y', 'start_station_code', 'end_station_code']
            dataset = dataset.drop(columns=columns_to_delete)

            dataset = dataset.rename(columns={
                'name_x': 'STARTSTATIONNAME',
                'latitude_x': 'STARTSTATIONLATITUDE',
                'longitude_x': 'STARTSTATIONLONGITUDE',
                'name_y': 'ENDSTATIONNAME',
                'latitude_y': 'ENDSTATIONLATITUDE',
                'longitude_y': 'ENDSTATIONLONGITUDE',
                'start_date': 'STARTTIMEMS',
                'end_date': 'ENDTIMEMS',
                'duration_sec': 'DURATIONSEC',
                'is_member': 'ISMEMBER'
            })

            dataset['STARTSTATIONARRONDISSEMENT'] = ''
            dataset['ENDSTATIONARRONDISSEMENT'] = ''

            dataset['STARTTIMEMS'] = pd.to_datetime(dataset['STARTTIMEMS']).astype('int64') // 10 ** 6
            dataset['ENDTIMEMS'] = pd.to_datetime(dataset['ENDTIMEMS']).astype('int64') // 10 ** 6

            datasets.append(dataset)

    final_dataset = pd.concat(datasets, ignore_index=True)
    final_dataset.drop_duplicates(inplace=True)
    final_dataset.sort_values(by='STARTTIMEMS', inplace=True)
    final_dataset.to_csv(f'../Data/{year}.csv', index=False)

for i in range(14, 22):
    print(f'20{i} starts')
    data_clean_year(f'20{i}')