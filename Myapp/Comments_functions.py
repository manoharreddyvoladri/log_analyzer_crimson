# def log_to_csv_in_session(request):
#     # Define the log file path and CSV file path
#     log_file_path = 'C:\\Users\\Crimson innovative\\Desktop\\Log Analysis Files\\Logfile_Operations-6-9-24\\Logfile_Operations-6-9-24\\Logfile_Operations\\LogFile_Container\\logdata.log'  # Update this path as needed

#     # Define a list of substrings to check for
#     substring_list = ["logPsm4Paramters", "TrdRobot", "EPU on-chip Temperature (Celsius): 23.00"]

#     # Define a function to parse each line of the log file
#     def parse_log_line(log_line):
#         # Regex to capture datetime and the entire log line after INFO
#         regex = r"(?P<datetime>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - Log:.*? (?P<log_level>INFO|WARNING|ERROR) (?P<event_template>.*)"


#         match = re.match(regex, log_line)
#         if match:

#             datetime_str = match.group('datetime')
#             event_template = log_line
            
#             # Find substring from the predefined list
#             substring = next((sub for sub in substring_list if sub in event_template), '')

#             return datetime_str, event_template, substring
#         return None, None, None

#     # Read log file, parse it, and write to CSV
#     csv_output = io.StringIO()
#     csv_writer = csv.writer(csv_output)
#     csv_writer.writerow(['Datetime', 'Event_Templete', 'Search_string'])

#     try:
#         with open(log_file_path, 'r') as log_file:
#             for line in log_file:
#                 datetime_str, event_template, substring = parse_log_line(line)
#                 if datetime_str:
#                     csv_writer.writerow([datetime_str, event_template, substring])
#     except FileNotFoundError:
#         return HttpResponse("Log file not found.", status=404)

#     # Move to the beginning of the StringIO object
#     csv_output.seek(0)

#     # Store the CSV content in session
#     request.session['csv_data'] = csv_output.getvalue()
#     print("csv data generated", request.session['csv_data'])
#     # Optionally, provide a response to confirm the data is stored
#     return HttpResponse("CSV data has been stored in session.")

# def Searching_CSV(request,user_name):
   
#     if request.method=='POST' and 'csv_file' in request.FILES:
#         print("calling")
#         csv_file = request.FILES['csv_file']
#         try:
#             df=pd.read_csv(csv_file)
#             print(df.head())
#             num_rows,num_columns=df.shape
#             data=df.to_dict(orient='records')
#             request.session['csv_data']=data
#             return render(request,'landing.html',{
#                 'user_name':user_name,
#                 'num_rows':num_rows,
#                 'num_columns':num_columns,
#                 'data':data            })
#             # return render(request,'Home_Page2.html',{
#             #     'num_rows':num_rows,
#             #     'num_columns':num_columns,
#             #     'data':data            })
#         except Exception as e:
#             return HttpResponse(f"Error processing search: {e}")
#     elif request.session['csv_data']:
#         try:
#             df=pd.DataFrame(request.session['csv_data'])
#             data=df.to_dict(orient='records')
#             return render(request,'landing.html',{
#                 'user_name':user_name,
#                 'data':data            })
#         except Exception as e:
#             return HttpResponse(f"Error processing search: {e}")
#     #return render(request,'Home_Page2.html')
#     return render(request,'landing.html',{'user_name':user_name,})
# def Visualization(request,user_name):
#     words_list=['crashpad','initialized','Recording','quality']
#     if not request.session['csv_data']:
#         return render(request,'visualization2.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
#     else:
#         df=pd.DataFrame(request.session['csv_data'])
#         data=df.to_dict(orient='records')
#         # words_list = [word.strip() for word in Default_Values.split(',')]
       
#         found_words = []
#         occurrences_list = []
#         alert_words=[]
#         alert_occurences=[]
#         try:
#             # df=pd.DataFrame(request.session['csv_data'])
            
#             for index, row in df.iterrows():
#                 # Check for each word in the EventTemplate column
#                 for word in words_list:
#                     if isinstance(row['EventTemplate'], str) and word.lower() in row['EventTemplate'].lower():
#                         # If the word is found, append it and the occurrences to the respective lists
#                         found_words.append(row['EventTemplate'])
#                         occurrences_list.append(int(row['Occurrences']))
#                         if int(row['Occurrences'])>10:
#                             alert_words.append(row['EventTemplate'])
#                             alert_occurences.append(int(row['Occurrences']))


#             print("Visulization Found Words1:####", found_words)
#             print("Visulization Occurrences List1:####", occurrences_list)
#             return render(request,'visualization2.html',{
#                 'user_name':user_name,
#                 'data':data,
#                  'default_words':found_words,
#                 'default_occurrences_list':occurrences_list,
#                 'default_alert_words':alert_words,
#                 'default_alert_occurences':alert_occurences
#                                 })
        
#             # return render(request,'Visualization.html',{
#             #     'data':data,
#             #     'word_counts':word_counts,
#             #      'num_words':len(word_counts)      })
#         except Exception as e:
#             print("exception occuing in search_words",{{e}})
#     return render(request,'visualization2.html',{
#                 'user_name':user_name,
#                 'data':""            })

# def search_words(request,user_name):
#     default_words_list=['crashpad','initialized','Recording','quality']
#     default_found_words = []
#     default_occurrences_list = []
#     print("calling this function")
#     if not request.session['csv_data']:
#         return render(request,'visualization2.html',{"message":"Please upload csv file in AddData.",'user_name':user_name,})
#     else:
#         df=pd.DataFrame(request.session['csv_data'])
#         data=df.to_dict(orient='records')
#     if request.method == 'POST':
#         # Get the input words
#         input_words = request.POST.get('serach_query')
#         print("inputwords are",input_words)
#         words_list = [word.strip() for word in input_words.split(',')]
       
#         found_words = []
#         occurrences_list = []
#         alert_words=[]
#         alert_occurences=[]
#         try:
#             # df=pd.DataFrame(request.session['csv_data'])
            
#             for index, row in df.iterrows():
#                 # Check for each word in the EventTemplate column
#                 for word in words_list:
#                     if isinstance(row['EventTemplate'], str) and word.lower() in row['EventTemplate'].lower():
#                         # If the word is found, append it and the occurrences to the respective lists
#                         found_words.append(row['EventTemplate'])
#                         occurrences_list.append(int(row['Occurrences']))
#                         if int(row['Occurrences'])>10:
#                             alert_words.append(row['EventTemplate'])
#                             alert_occurences.append(int(row['Occurrences']))
#                 for word in default_words_list:
#                     if isinstance(row['EventTemplate'], str) and word.lower() in row['EventTemplate'].lower():
#                         # If the word is found, append it and the occurrences to the respective lists
#                         default_found_words.append(row['EventTemplate'])
#                         default_occurrences_list.append(int(row['Occurrences']))
#                         # if int(row['Occurrences'])>10:
#                         #     alert_words.append(row['EventTemplate'])
#                         #     alert_occurences.append(int(row['Occurrences']))

#             print("Visulization Found Words1:####", default_found_words)
#             print("Visulization Occurrences List1:####", occurrences_list)
#             # print("searcb_words found_words:", found_words)
#             # print("serach_words occurence words:", occurrences_list)
#             return render(request,'visualization2.html',{
#                 'user_name':user_name,
#                 'data':data,
#                  'Found_Words':found_words,
#                 'Occurrences_List':occurrences_list,
#                 'alert_words':alert_words,
#                 'alert_occurences':alert_occurences,
#                 'default_words':default_found_words,
#                 'default_occurrences_list':default_occurrences_list,
#                                 })
#             # return render(request,'Visualization.html',{
#             #     'data':data,
#             #     'word_counts':word_counts,
#             #      'num_words':len(word_counts)      })
#         except Exception as e:
#             print("exception occuing in search_words",{{e}})
#         # Load your CSV file into a DataFrame
#         # df = pd.read_csv('D:\\Idur_Sir_Project\\Virtual_Environment\\CSV_Files\\ex.csv')  # Specify the correct path

#         # Initialize a dictionary to hold word counts
        

#         return render(request,'visualization2.html',{
#             'user_name':user_name,
#                 'data':data            })

#     return render(request,'visualization2.html',{
#         'user_name':user_name,
#                 'data':data            })

#     return render(request,'visualization2.html',{
#         'user_name':user_name,
#                 'data':data            })
# def Read_Logfile(request):
#     if request.method == 'POST' and request.FILES['log_file']:
#         log_file = request.FILES['log_file']
#         content = log_file.read().decode('utf-8')
        
#         # Count the occurrences of each word
#         words = re.findall(r'\b\w+\b', content.lower())
#         word_counts = Counter(words)
#         print(word_counts)
#         # return render(request, 'display_word_counts.html', {
#         #     'word_counts': word_counts
#         # })
#     return HttpResponse("success")
# # views.py


# def generate_chart():
#     plt.figure(figsize=(15, 6))
#     # Example chart (replace with your own chart code)
#     plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#     plt.title('Example Chart')

#     # Save the chart to a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
    
#     # Encode the image to base64 to embed in HTML
#     string = base64.b64encode(buf.read())
#     uri = urllib.parse.quote(string)
    
#     return uri

# def chart_view(request):
#     chart_uri = generate_chart()
#     return render(request, 'chart.html', {'chart_uri': chart_uri})
