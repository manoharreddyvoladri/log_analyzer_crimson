from django.urls import path
from . import views

urlpatterns=[
    path('',views.User_Login,name='User_Login'),
    path('Register',views.Register,name='Register'),
    path('Dashboard/<str:user_name>',views.Dashboard,name='Dashboard'),
    path('Get_Strings_Logs/<str:user_name>',views.Get_Strings_Logs,name='Get_Strings_Logs'),
    path('Visualization_Task/<str:user_name>/<str:calling_request>',views.Visualization_Task,name='Visualization_Task'),
    path('Prediction_Task/<str:user_name>/<str:calling_request>',views.Prediction_Task,name='Prediction_Task'),
    path('Result_Forecast/<str:user_name>/', views.Result_Forecast, name='Result_Forecast'),
    # path('Visualization/<str:user_name>',views.Visualization,name='Visualization'),
    # path('Searching_CSV/<str:user_name>',views.Searching_CSV,name='Searching_CSV'),
    # path('search_words/<str:user_name>',views.search_words,name='search_words'),
    path('User_Forgot_Password',views.User_Forgot_Password,name='User_Forgot_Password'),
    path('User_Verfiy_Otp',views.User_Verfiy_Otp,name='User_Verfiy_Otp'),
    path('User_Logout',views.User_Logout,name='User_Logout'),
    path('Resend_Otp',views.Resend_Otp,name='Resend_Otp'),
    # path('Read_Logfile',views.Read_Logfile,name='Read_Logfile'),
    # path('chart_view',views.chart_view,name='chart_view'),
    path('Soring_Word_Wise',views.Soring_Word_Wise,name='Soring_Word_Wise'),
    path('Sorting_Data/<str:user_name>',views.Sorting_Data,name='Sorting_Data'),
    path('Healthy/<str:user_name>',views.Healthy,name="Healthy"),
    path('Show_Warning/<str:user_name>',views.Show_Warning,name='Show_Warning'),
    path('Show_Errors/<str:user_name>',views.Show_Errors,name='Show_Errors'),
    path('All_Data/<str:user_name>',views.All_Data,name='All_Data'),
    path('Add_New_Keyword/<str:user_name>',views.Add_New_Keyword,name='Add_New_Keyword'),
    path('Meter_values',views.Meter_values,name='Meter_values'),
    path('Warning_Loop',views.Warning_Loop,name='Warning_Loop'),
    path("Error_Loop",views.Error_Loop,name="Error_Loop"),
    path("All_Data_Loop",views.All_Data_Loop,name="All_Data_Loop"),
    path("Visualization_Loop",views.Visualization_Loop,name="Visualization_Loop"),
    path("for_checking",views.for_checking,name="for_checking"),
    path('train-model/<str:user_name>/', views.Train_Model, name='Train_Model'),
    #   path('train_model/<str:user_name>/', Train_Model, name='Train_Model')
    path('get_combined_forecast/', views.get_combined_error_forecast, name='get_combined_error_forecast'),
    # path('debug-session/', views.debug_session, name='debug_session'),
    # path('clear-session/', views.clear_session, name='clear_session'),
    path('Show_Info/<str:user_name>', views.Show_Info, name='Show_Info'),

    
    
    
]