# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:38:59 2022

@author: gusze
"""


#import re
import requests
import csv
from bs4 import BeautifulSoup
import os
import time
import numpy as np
import pickle
import bz2
import pandas as pd
import shutil


def convert_text_list_to_float_array(text_list, missing=np.nan):
    N = len(text_list)
    res = np.zeros(N,dtype=np.float32)
    for i in range(N):
        try:
            val = float(text_list[i])
        except:
            try:
                val = ( float(text_list[i-1]) + float(text_list[(i+1)%N]) )/2
            except:
                val = missing
        res[i] = val
    return res

def pickle_dump(fname, var, use_bz2=False, make_backup=True):
    if make_backup:
        if os.path.exists(fname):
            shutil.copy(fname, fname+'.bak')
    if use_bz2:
        with bz2.BZ2File(fname, 'wb') as pickle_out:
            pickle.dump(var, pickle_out, protocol=2)    
    else:
        with open(fname,'wb') as pickle_out:
            pickle.dump(var, pickle_out)

def pickle_load(fname, use_bz2=False):
    if use_bz2:
        with bz2.BZ2File(fname, 'rb') as pickle_in:
            var = pickle.load(pickle_in, encoding='bytes')
    else:
        with open(fname,'rb') as pickle_in:
            var = pickle.load(pickle_in)
    return var


def append_or_create_list_in_dict(dictionary, dict_label, value):
    if dict_label in dictionary.keys():
        dictionary[dict_label].append(value)
    else:
        dictionary[dict_label] = [value]

def soup_response_links(url):
    response = requests.get(url, '')
    soup = BeautifulSoup(response.content, 'lxml')
    return soup.find_all("a")  # Find all urls

def format_url_from_soup_item(soup_item):
    global url_top
    url = soup_item.get('href').strip()
    if not (url_top in url):
        url = url_top + url
    return url

def clean_text(text):
    to_remove = [' is ', ' are ', ' of ', ' for ', ':', ';', ',', '(', ')', '[', ']', '   ', '  ']
    new_text = text + ''
    for w in to_remove:
        new_text = new_text.replace(w,' ')
    return new_text.lower().strip()

def is_angular_coord(words):
    return ('°' in words[0]) and ('\'' in words[1]) and ( ('\"' in words[2]) or ('\'\'' in words[2]) )

def angular_coord_words_to_deg(words):
    deg = float(words[0].split('°')[0].replace('°',''))
    arcmin = float(words[1].split('\'')[0].replace('\'',''))
    arcsec = float( (words[2].split('\"')[0].replace('\"','')).split('\'\'')[0].replace('\'\'','') )
    deg_full = deg + arcmin/60 + arcsec/3600
    return deg_full
    

####################################
#Get city links

def process_link(soup_item, region='', country='', state='', depth_level=0):
    global city_links, cities, states, countries, regions #access global lists
    url = format_url_from_soup_item(soup_item)
    if (depth_level==0) and ('/weather' in url) and ( ('&regionname=' in url) or ( ('Antarctica' in url) and not ('cityname' in url) ) ): #handle Antarctica edge case
        region = soup_item.getText().strip().title()
        if (region in regions): return #no need to do a region 
        if (region=='Arctic'): return #skip the North Pole
        print('\nProcessing '+region)
        depth_level = 1
        for item in soup_response_links(url):
            if url != format_url_from_soup_item(item): #to avoid self-referencing links
                process_link(item, region=region, depth_level=depth_level)
        time.sleep(2)
    #Countries
    elif (depth_level==1) and ('/weather' in url) and ('&name=' in url):
        depth_level = 2
        country = soup_item.getText().strip()
        if (country in countries): return #no need to do a country twice
        for item in soup_response_links(url):
            process_link(item, region=region, country=country, depth_level=depth_level)
    #States
    elif (depth_level==2) and ('&statename=' in url):
        depth_level = 3
        state = soup_item.getText().strip()
        for item in soup_response_links(url):
            process_link(item, region=region, country=country, state=state, depth_level=depth_level) 
    #Cities
    elif (depth_level>0) and ('&cityname=' in url):
        if (region=='Antarctica'): country = region #handle edge case
        city = soup_item.getText().strip()
        url += '&units=metric'
        full_key = (country + '/' + state + '/' + city).replace('//','/')
        if not (full_key in full_keys): #Check if this is a double:
            city_links.append(url)
            cities.append(city)
            states.append(state)
            countries.append(country)
            regions.append(region)
            full_keys.append(full_key)
            print('\t' + (region + '/' + country + '/' + state + '/' + city +' added').replace('//','/') )


link_fname = 'weatherbase_city_links.csv'
url_top = 'https://www.weatherbase.com'
regions = []; states = []; countries = []; cities = []; city_links = []; full_keys = []
if not os.path.exists(link_fname):
    top_urls = soup_response_links(url_top)
    # Cycle through all urls
    for item in top_urls:
        process_link(item)
    # open the file in the write mode
    with open(link_fname, 'w', encoding="utf8") as f:
        writer = csv.writer(f)
        for i in range(len(city_links)):
            writer.writerow( [regions[i], countries[i], states[i],  cities[i], city_links[i], full_keys[i]] )
else:
    with open(link_fname, 'rt', encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row):
                regions.append(row[0]); countries.append(row[1]); states.append(row[2]); cities.append(row[3]); city_links.append(row[4]); full_keys.append(row[5]); 
    print('City URLs loaded from '+link_fname)

regions = np.array(regions); countries = np.array(countries); states = np.array(states); cities = np.array(cities); city_links = np.array(city_links); full_keys = np.array(full_keys);

# ####################################
# #Get coordinates of cities from Google serches
# pickle_name = 'city_coord.p'
# get_coords=True
# redo_nans=False
# wait_time_sec=1.0

# if not os.path.exists(pickle_name):
#     city_coord = {}; city_lat_long = {}
# else:
#     temp = pickle_load(pickle_name)
#     city_coord = temp[0]; city_lat_long = temp[1];
#     print('City coordinates loaded from '+pickle_name)
# if get_coords:
#     N_city = len(cities); N_loaded = len(city_coord)
#     for num, (city, country) in enumerate(zip(cities,countries)):
#         key = country+'/'+city
#         skip_this=True
#         if not (key in city_coord.keys()): 
#             skip_this = False
#         elif redo_nans and ( np.isnan(city_coord[key][0]) or np.isnan(city_coord[key][1]) ): 
#             skip_this = False
#         if not skip_this:
#             time.sleep(wait_time_sec)
#             url = 'https://www.google.com/search?q=' + city + '+' + country + '+gps'
#             response = requests.get(url, '')
#             soup = BeautifulSoup(response.content, 'lxml')
#             text = soup.getText().lower()
#             if 'unusual traffic from your computer network' in text:
#                 print('\n \n Google refused requests!\n Try with larger wait time...\n')
#                 break
#             #Clean text
#             text = clean_text(text)
#             words = text.split(' ')
#             city_coord[key] = np.array([np.nan, np.nan])
#             city_lat_long[key] = np.array([np.nan, np.nan])    
            
#             lat_guesses = []
#             long_guesses = []
#             for i,w in enumerate(words):
#                 if is_angular_coord(words[(i+1):(i+4)]):
#                     #print(words[i:i+10]) 
#                     try:
#                         deg_full = angular_coord_words_to_deg(words[(i+1):(i+4)])
#                         radian = deg_full * np.pi/180 
#                         #Convert to spherical coordinates
#                         direction = ''
#                         if ( (w == 'e') or (w == 'w') ) and ( (words[i-4] == 's') or (words[i-4] == 'n') ):
#                             direction = w
#                         elif ( (w == 'n') or (w == 's') ) and ( (words[i+4] == 'e') or (words[i+4] == 'w') ):
#                             direction = w
#                         elif ('north' in words[i+4]) or (words[i+4] == 'n'):
#                             direction = 'n'
#                         elif ('south' in words[i+4]) or (words[i+4] == 's'):
#                             direction = 's'
#                         elif ('west' in words[i+4])or ('w.' in words[i+4])  or (words[i+4] == 'w'):
#                             direction = 'w'
#                         elif ('east' in words[i+4]) or ('e.' in words[i+4]) or  (words[i+4] == 'e'):
#                             direction = 'e' 

                            
#                         if direction == 'n':
#                             coord_index = 0
#                             radian = np.pi/2 - radian
#                         elif direction == 's':
#                             coord_index = 0
#                             radian = np.abs(radian) + np.pi/2
#                             deg_full = -np.abs(deg_full)
#                         elif direction == 'w':
#                             coord_index = 1
#                             radian = -np.abs(radian)
#                             deg_full = -np.abs(deg_full)
#                         elif direction == 'e':
#                             coord_index = 1
#                         else:
#                             coord_index = 3 
                        
#                         if coord_index==0:
#                             lat_guesses.append([deg_full,radian])
#                         elif coord_index==1:
#                             long_guesses.append([deg_full,radian])   
#                         # if coord_index!=3:
#                         #     print(direction, coord_index, deg_full, words[i:(i+5)])         
#                     except:
#                         pass
#                 if i > (len(words)-5):
#                     break
     
#             if len(lat_guesses):
#                 #print(np.array(lat_guesses)[:,0])
#                 city_lat_long[key][0] = np.median(lat_guesses,axis=0)[0]
#                 city_coord[key][0] = np.median(lat_guesses,axis=0)[1]
#             if len(long_guesses):
#                 #print(np.array(long_guesses)[:,0]) 
#                 city_lat_long[key][1] = np.median(long_guesses,axis=0)[0]
#                 city_coord[key][1] = np.median(long_guesses,axis=0)[1]
            
#             #Try less trustworthy methods if we still have no guesses for one
#             if np.isnan(city_coord[key][0]) or np.isnan(city_coord[key][1]):
#                 print("GPS coordinates not found for "+key+' trying sketchier methods')
#                 for i,w in enumerate(words):
#                     if ('coordinates' in w):
#                         temp_w = words[i+1].replace('.gps','').split(',')
#                         if (len(temp_w)==2):
#                             try:
#                                 if np.isnan(city_coord[key][0]):
#                                     deg_full = float(temp_w[0])
#                                     radian = (90 - deg_full) * np.pi/180
#                                     city_coord[key][0] = radian
#                                     city_lat_long[key][0] = deg_full
#                                 if np.isnan(city_coord[key][1]):
#                                     deg_full = float(temp_w[1])
#                                     radian = deg_full * np.pi/180
#                                     city_coord[key][1] = radian
#                                     city_lat_long[key][1] = deg_full
#                             except:
#                                 pass
#                     if ('latitude' in w) or ('longitude' in w):
#                         temp_w = words[i+1].replace('.gps','').strip()
#                         deg_full = np.nan
#                         try:
#                             deg_full = float(temp_w)
#                         except:
#                             pass
#                         if is_angular_coord(words[(i+1):(i+4)]):
#                             try:
#                                 deg_full = angular_coord_words_to_deg(words[(i+1):(i+4)])
#                             except:
#                                 pass
#                         if np.isfinite(deg_full):
#                             radian = deg_full * np.pi/180
#                             if ('longitude' in w) and np.isnan(city_coord[key][1]):
#                                 city_coord[key][1] = radian
#                                 city_lat_long[key][1] = deg_full
#                             elif ('latitude' in w) and np.isnan(city_coord[key][0]):
#                                 city_coord[key][0] = radian
#                                 city_lat_long[key][0] = deg_full

#                     #Terminate if we could not find a good guess
#                     if i > (len(words)-5):
#                         print("Failed to find GPS coordinates for "+key)
#                         # print(url)
#                         # print(words)
#                         # print(city_lat_long[key])
#                         # kkk
#                         break
#                     if np.isfinite(city_coord[key][0]) and np.isfinite(city_coord[key][1]):
#                         break           
#             if np.isfinite(city_coord[key][0]) and np.isfinite(city_coord[key][1]):
#                 print(key+' %d / %d Latitude %g° Longitude %g°'%(num+1,N_city,city_lat_long[key][0],city_lat_long[key][1])) 

#         if (num>N_loaded) and (not (num%100)):
#             pickle_dump(pickle_name,[city_coord,city_lat_long]) #save in case we need to restart
#     pickle_dump(pickle_name,city_coord)

def child_key(parent_key, i):
    return parent_key + ', month ' + str(i)
                    
###################################
#List the metrics we want to read (not all cities have all of these)
parent_keys = ['Average Temperature', 'Average High Temperature', 'Average Low Temperature',
 'Average Precipitation', 'Average Number of Days With Precipitation',
 'Average Length of Day', 'Average Number of Days Above 90F/32C',
 'Average Number of Days Below 32F/0C', 'Average Number of Days With Snow',
 'Average Relative Humidity', 'Average Morning Relative Humidity', 'Average Evening Relative Humidity',
 'Average Daily Sunshine', 'Average Wind Speed']
#Init
columns = {'Key' : [],
           'City' : [],
           'State' : [],
           'Country' : [],
           'Region' : [],
           'Latitude': [],
           'Longitude': [],
           'Elevation': []
           }
#Add empty monthly data columns
for parent_key in parent_keys:
    for i in range(12):
        columns[child_key(parent_key, i+1)] = []
column_labels = columns.keys()

####################################
#Get data for each city
get_missing_city_data = True
#wait_time_sec=0 #1.0
pickle_name = 'city_climate_columns.p'
respone_pickle_name = 'city_climate_web_response.p'
response_dict = {}

N_processed = 0
if os.path.exists(pickle_name):
    columns = pickle_load(pickle_name)
    print('City column data loaded from '+pickle_name)
N_loaded = len(columns['City']); N_city = len(cities);
if get_missing_city_data:
    for ind, (city, country, url, key) in enumerate(zip(cities,countries,city_links, full_keys)):
        alt_key = country+'/'+city
        if alt_key!=key: #let's correct from previous convention where key was country + city
            N_dup = np.sum( (cities==city) &(countries==country) )
            if alt_key in response_dict.keys(): #fix responses
                if not N_dup: #it means we can use the data, this is just a simple renaming
                    response_dict[key] = response_dict[alt_key].copy()
                del response_dict[alt_key] #erase old on
            if alt_key in columns['Key']: #fix columns
                row_num = np.argmax(np.array(columns['Key'])==alt_key)
                if not N_dup: #it means we can use the data, this is just a simple renaming
                    columns['Key'][row_num] = key
                else: #we need to erase this
                    for label in columns.keys():
                        del columns[label][row_num]
        if not (key in columns['Key']): #only do cities we haven't done before
            if (len(response_dict)==0) and os.path.exists(respone_pickle_name):
                response_dict = pickle_load(respone_pickle_name, use_bz2=True)
                print('Downloaded web data loaded from '+respone_pickle_name)
            if not (key in response_dict.keys()):
                response = requests.get(url, '')
                response_dict[key] = response
            else:
                response = response_dict[key]
            soup = BeautifulSoup(response.content, 'lxml')
            if len(soup.select('.data')):
                print('Processing '+key+' %d / %d'%(ind+1,N_city))
                N_processed += 1
                #Get longitude, latitude and elevation
                latitude = np.nan; longitude = np.nan; elevation = np.nan; #init
                for item in soup.find_all("meta"):
                    if item.get('itemprop')=='latitude':
                        latitude = float( item.get('content') )
                    if item.get('itemprop')=='longitude':
                        longitude = float( item.get('content') )
                    if item.get('itemprop')=='elevation':
                        elevation_text = item.get('content')
                        elevation = float(elevation_text.split(' ')[0].strip())
                        if ('feet' in elevation_text) or ('feet' in elevation_text):
                            elevation *= 0.3048 #let's use normal units 
                N_labels = len(soup.select('.h4line'))
                data_labels = [soup.select('.h4line')[i].getText() for i in range(N_labels)]; data = {}
                for i, label in enumerate(data_labels):
                    data[label] = convert_text_list_to_float_array( [soup.select('.data')[j].getText() for j in range(i*13+1,i*13+13)] ) #get monthly averages
                #Find requested keys
                for label in column_labels:
                    label_split = label.split(', month')
                    parent_label = label_split[0].strip()
                    if len(label_split)==2:
                        labelnum = int(label.split('month')[1].strip())
                    if label=='Key':
                        append_or_create_list_in_dict(columns, label, key)
                    elif label=='City':
                        append_or_create_list_in_dict(columns, label, city)
                    elif label=='State':
                        append_or_create_list_in_dict(columns, label, states[ind])
                    elif label=='Country':
                        append_or_create_list_in_dict(columns, label, country)
                    elif label=='Region':
                        append_or_create_list_in_dict(columns, label, regions[ind])
                    elif label=='Latitude':
                        append_or_create_list_in_dict(columns, label, latitude)
                    elif label=='Longitude':
                        append_or_create_list_in_dict(columns, label, longitude)
                    elif label=='Elevation':
                        append_or_create_list_in_dict(columns, label, elevation)
                    elif parent_label in data.keys():
                        append_or_create_list_in_dict(columns, label, data[parent_label][labelnum-1])
                    else: #we don't have this
                        append_or_create_list_in_dict(columns, label, np.nan)
                #time.sleep(wait_time_sec)
            else:
                print("No data for "+key+" at link: "+url)
                print(soup)
                exit()
        if (N_processed>0) and (not (N_processed%1000)):
            print('Saving all downloaded data, in case the run is interrupted...')
            pickle_dump(pickle_name,columns) #save in case we need to restart
            pickle_dump(respone_pickle_name, response_dict, use_bz2=True) #save in case we need to restart
            print('Done')
    print('Saving all downloaded data...')
    pickle_dump(pickle_name,columns) #save in case we need to restart 
    pickle_dump(respone_pickle_name, response_dict, use_bz2=True) #save in case we need to restart
    print('Done')

#Convert to numpy
for key in columns.keys():
    columns[key] = np.array(columns[key])

    
#Create dataframe
df = pd.DataFrame.from_dict(columns)
df.set_index('Key',inplace=True)
df.sort_index(inplace=True)
#Save original dataframe
df_fname = 'city_climate_dataframe_no_cleaning'
df.to_pickle(df_fname + '.p')
df.to_csv(df_fname + '.csv')
print('Original dataframe saved to %s and %s'%(df_fname + '.p', df_fname + '.csv', ) )

################################################
#Clean data
print('Starting cleaning...')

#Fix NaN in numbers where possible
fixable_parent_keys = np.array([key for key in parent_keys if ('Number of Days' in key) or ('Precipitation' in key)])

def nan_cleaner(df_row):
    df_row_cleaned = df_row.copy()
    for parent_key in fixable_parent_keys:
        data = np.array( [df_row[child_key(parent_key, j)] for j in range(1,13)] ) #get the monthly data
        finite_num = np.sum(np.isfinite(data))
        nan_num = np.sum(np.isnan(data))
        #print(nan_num,finite_num)
        if (finite_num or ('Precipitation' in parent_key) )and nan_num: #we have a mix of NaNs and non-NaNs, so it is measured but missing for some months, very likely those values are zeros. In case of Precipitation some areas really have essentially zero, so we fill those
            data[np.isnan(data)] = 0
            for j in range(1,13):
                df_row_cleaned.at[child_key(parent_key, j)] = data[j-1]
    return df_row_cleaned

def humidity_guesser(df_row):
    df_row_cleaned = df_row.copy()
    data = {}
    humidity_keys = ['Average Morning Relative Humidity', 'Average Evening Relative Humidity', 'Average Relative Humidity']
    for parent_key in humidity_keys:
        if not(child_key(parent_key, 1) in df_row.keys()): #catch in case we don't even have NaNs for these
            return df_row_cleaned
        data[parent_key] = np.array( [df_row[child_key(parent_key, j)] for j in range(1,13)] )
    if not np.any(np.isfinite(data['Average Relative Humidity'])): #This line is missing
        if np.any(np.isfinite(data['Average Morning Relative Humidity'])) and np.any(np.isfinite(data['Average Evening Relative Humidity'])): #we have th morning and evening data though
            humidity_guess = (data['Average Morning Relative Humidity'] + data['Average Evening Relative Humidity']) / 2
            for j in range(1,13):
                df_row_cleaned.at[child_key('Average Relative Humidity', j)] = humidity_guess[j-1]
    return df_row_cleaned

def shift_months_for_southern_seasons(df_row):
    #Shift southern hemispshere data by 6 months
    if df_row['Latitude']>=0:
        return df_row
    else:
        df_row_shifted = df_row.copy()
        for key in df_row.keys():
            if 'month' in key:
                num = int( key[-2:].strip() )
                if num < 7: #no need to do it twice
                    switch_key = (key[:-2] + ' ' + str( (num - 1 + 6)%12 +1 )).strip().replace('  ',' ')
                    df_row_shifted.at[key] = df_row[switch_key]
                    df_row_shifted.at[switch_key] = df_row[key]
        return df_row_shifted


df_cleaned = df.copy()
print('\tEstimating missing average humidity data')
df_cleaned = df_cleaned.apply(humidity_guesser,axis=1)

print('\tFixing NaNs, if possible')
df_cleaned = df_cleaned.apply(nan_cleaner,axis=1)
print('\tShifting Southern Hemisphere data')
df_cleaned = df_cleaned.apply(shift_months_for_southern_seasons,axis=1)

#Throw away data that is useless
print('\t Dropping data that is missing vital keys')
vital_keys = np.array([ col for col in np.array(df.columns) if ('Average Temperature' in col) or ('Average Precipitation' in col) or ('Average Relative Humidity' in col) ])
N_orig = len(df_cleaned)
for vital_key in vital_keys:
    N_items = len(df_cleaned)
    df_cleaned.dropna(axis=0, subset=[vital_key], inplace=True)
    N_drop = N_items - len(df_cleaned)
    if N_drop:
        print('\t\t%d out of %d records dropped for missing %s,\n\t\t\t Representing %g %% of remaining data'%(N_drop, N_items, vital_key, 100*N_drop/N_items ))

#Remove columns that are almost all NaNs
print("\tFinding NaN fraction of columns")
NaN_threshold = 0.9
bad_columns = []; bad_parent_keys = []
for parent_key in parent_keys:
    min_NaN_frac = np.min( [ df_cleaned[child_key(parent_key, i)].isna().sum()/len(df_cleaned) for i in range(1,13)] )
    if min_NaN_frac:
        print("\t\t%s: %g %% of data missing for all months"%(parent_key, min_NaN_frac*100))
    if min_NaN_frac>NaN_threshold:
        #print(parent_key, [ df_cleaned[child_key(parent_key, i)].isna().sum()/len(df_cleaned) for i in range(1,13)])
        bad_columns += [ child_key(parent_key, i) for i in range(1,13)]
        bad_parent_keys.append(parent_key)    
print("\tRemoving columns with above %g %% fraction of NaNs, these are:\n\t"%(NaN_threshold*100),list(bad_parent_keys))
df_cleaned.drop(labels=bad_columns,axis=1, inplace=True)




#Save cleaned dataframe
df_fname = 'city_climate_dataframe'
df_cleaned.to_pickle(df_fname + '.p')
df_cleaned.to_csv(df_fname + '.csv')
print('Cleaning done, summary:\n\t %d rows kept out of %d\n\t %d columns kept out of %d'%(len(df_cleaned), len(df),len(df_cleaned.keys()), len(df.keys())) )
print('Dataframe saved to %s and %s'%(df_fname + '.p', df_fname + '.csv', ) )

