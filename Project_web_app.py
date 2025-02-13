import pickle
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import sklearn
from sklearn.ensemble import AdaBoostClassifier

loaded_model = load("D:/University Final Year Project/FYP_ML_model.joblib")

#define function to to get neighbours and its distances
#from the passed node
def get_neighbours (v, Graph_nodes ):
  if v in Graph_nodes:
    return Graph_nodes[v]
  else:
    return None

#def a function to add block on the nodes
def Map_B(Block):


  Map[str(Block)] = None

  return Map

# A Star algo
def AStarAlgo(start_node, stop_node, Graph_nodes):

  open_set = {start_node}
  closed_set = set()
  g = {} #store distance from starting node
  parents = {} #parent keep an adjancency map of all nodes

  #distance from starting node to itself is zero
  g[start_node] = 0
  #starting node is root node it has no parent node
  #so start node is set to its own parent node
  parents[start_node] = start_node

  while len(open_set) > 0:
    n = None

    #node with the lowest f is found
    for v in open_set:

      if n == None or g[v] + heuristic(str(v)) <g[n] + heuristic(str(n)):
         n = v


    if n == stop_node or Graph_nodes[n] == None:
      pass

    else:
      for (m, weights) in get_neighbours(n, Graph_nodes):
        #node m not in the first and the set are added to the first set
        #n is set to its parent
        if m not in open_set and m not in closed_set:
          open_set.add(m)
          parents[m] = n
          g[m] = g[n] + weights

          #from each node m, compare its distance from start
          #from start to send node

        else:
          if g[m] > g[n] + weights:
              #update g(m)
              g[m] = g[n] + weights
              #change parent from n to m
              parents[m] = n

              #if m is in closed det remove and it to the open set
              if m in closed_set:
                closed_set.remove(m)
                open_set.add(m)



    if n == None:
      print("Path does not exist")
      return None

    #if current node is the stop node
    # then we begin reconstructing the path from thst node to the start node

    if n == stop_node:
      path = []

      while parents[n] != n:
        path.append(n)
        n = parents[n]

      path.append(start_node)

      path.reverse()

      #print("path found {}".format(path))
      return path

      #remove n from the open list and add it to the closed list
      #because all of his neighbours were inspected

    open_set.remove(n)
    closed_set.add(n)
  print("path does not exist")
  return None


#for simplicity we will consider heuristic distances given
#this function gives the heuristic function for all nodes

def heuristic(n):
  H_dist = {
      "1": 11,
    "2": 22,
    "3": 12,
    "4": 21,
    "5": 32,
    "6": 21,
    "7" : 33,
    "8" : 43,
    "9": 23,
    "10": 43,
    "11": 34,
    "12": 36,
    "13": 26,
    "14" : 30,
    "15":28,
    "16": 31,
    "17": 200,
    "18": 29,
    "19": 25,
    "20": 20,
    "21": 99,
    "22": 100
 
  }

  return H_dist[n]


#Describe your graph here

Map = {
    "1": [("2", 120), ("19", 500)],
    "2": [("3", 110), ("1", 120)],
    "3": [("4", 210), ("19", 460), ("2",110)],
    "4": [("5", 450), ("3", 210)],
    "5": [("6", 450), ("4",450)],
    "6": [("7", 190), ("15", 650), ("5", 450)],
    "7": [("8", 280), ("9" , 400), ("6", 190)],
    "8": [("9", 220), ("7", 280)],
    "9": [("10", 80), ("8",220), ("7", 400)],
    "10": [("11", 350), ("15", 200), ("9",80)],
    "11": [("12", 450),("16", 150), ("10", 350)],
    "12": [("13", 110 ), ("11",450)],
    "13": [( "14", 250), ("17", 350), ("12", 110)],
    "14": [( "13", 250), ("22", 230)],
    "15": [("10", 200), ("16", 350), ("6", 650)],
    "16": [("11", 150), ("17", 280), ("21", 100), ("15", 350)],
    "17": [("13", 350), ("18", 350), ("16", 280)],
    "18": [("22", 350), ("22", 750)],
    "19": [("20", 400), ("3", 460), ("1", 500)],
    "20": [("21", 1100), ("19", 400)],
    "21": [("16", 100), ("20", 1100)],
    "22": [( "14", 230), ("18", 350), ("18", 750)]

}

#Dictionary with names of the nodes.
names = {
    '1' : "Islamia Road(1)",
    '2' : "Deans Gate(2)",
    '3' : "FC Chowk(3)",
    '4' : "Ajab Khan Afridi Road(4)",
    '5' : "Railway Road(5)",
    '6' : "Railway Road(6)",
    '7' : "Dabgari Garden(7)" ,
    '8' : "Kohat Road(8)" ,
    '9' : "Dabgari Garden(9)",
    '10':"Kohat Road(10)",
    '11' : "Cinema Road(11)",
    '12' : "Cinema Road(12)",
    '13' : "Hakeem Ullah Jan Road(13)",
    '14' : "Lady Reading Hospital Emergency Gate(14)" ,
    '15' : "Railway Road and Shoba Bazar(15)",
    '16' : "Khyber Bazar(16)",
    '17' : "Qissa khawai Road(17)",
    '18' : "Soekarno Road(18)",
    '19' : "Sunehri Masjid Road(19)",
    '20' : "Sunehri Masjid Road(20)",
    '21' : "Bjori Road(21)",
    '22' : "Hakeem Ullah Jan Road(22)"
    
}


# function to call a star algo and enter starting and end location.
def final_astar(a,b, Block):

  path = AStarAlgo(a, b, Graph_nodes = Map_B(Block) )

  return path

def final_astar_noblock(a,b, Map):

  path_noblock = AStarAlgo(a, b, Graph_nodes = Map )

  return path_noblock

#trace the shortest path names
def path_names(path):

   path_found = []
   for route in path:
     pf = names[str(route)]
     path_found.append(pf)
   return path_found

#find the coordinates for shortest path
def shortest_path_coordinates(df1, path):

  shortest_path = [ ]
  for i in path:
      for j in df1.itertuples():
        if int(i) == j.points:
          shortest_path.append([j.Latitude, j.Longitude])
  return(shortest_path)

#map of the shortest path
def shortest_path_map(df1, path, Block):

    map_ant_route = folium.Map(location=[34.007285, 71.560445], zoom_start=14.4)

    coordinate = None

    for i in df1.itertuples():
        if int(Block) == int(i.points):
            coordinate = (i.Latitude, i.Longitude)

    if coordinate is not None:
        folium.CircleMarker(location= coordinate,
                            radius=10,
                            color='red',
                            fill=True,
                            fill_color='red',
                            fill_opacity=1).add_to(map_ant_route)


    for i in shortest_path_coordinates(df1, path):
        for j in df1.itertuples():
            if i == [j.Latitude, j.Longitude]:
                folium.Marker(location=(j.Latitude, j.Longitude),
                              popup=j.Name,
                              icon=plugins.BeautifyIcon(number=j.points,
                                                        border_color="blue",
                                                        border_width=1,
                                                        text_color="red",
                                                        inner_icon_style="margin-top:opx")).add_to(map_ant_route)

    plugins.AntPath(shortest_path_coordinates(df1, path), color="green").add_to(map_ant_route)

    return map_ant_route

#map with the number icons
def number_icon_map(df1):
    map_plot_icon = folium.Map(location=[34.007285, 71.560445], zoom_start=14.4)

    for i in df1.itertuples():
        folium.Marker(location=(i.Latitude, i.Longitude),
                      popup=i.Name,
                      icon=plugins.BeautifyIcon(number=i.points,
                                                border_color="blue",
                                                border_width=1,
                                                text_color="red",
                                                inner_icon_style="margin-top:opx")).add_to(map_plot_icon)

    return map_plot_icon


#map of the selected path
def selected_path_map(df):
    lats_longs = df.drop(columns=["points:", "Name:"])
    route_lats_longs = lats_longs.to_numpy()

    map_plot_route = folium.Map(location=[34.007285, 71.560445], zoom_start=14.4)
    folium.PolyLine(route_lats_longs, color="purple").add_to(map_plot_route)

    return map_plot_route

#Adding color to the data frame
def add_color(value):
    Prob = value["Probability of Congestion"]
    Pred = value["Prediction"]

    if Prob <= 0.25 and Pred == 0:
        color = "lightgreen"
    elif 0.26 <= Prob <= 0.50 and Pred == 0:
        color = "yellow"
    elif 0.51 <= Prob <= 0.75 and Pred == 1:
        color = "orange"
    else:
        color = "red"
    return [f"background-color: {color}" for _ in value]

#Function to display result
def prediction_result(data, loaded_model):

    Hourly_Data = data.groupby("Hours")[["Flowrate", "Velocity", "Density"]].mean()
    data_scaled = scaler.fit_transform(Hourly_Data)

    Predictions = loaded_model.predict(data_scaled)
    Predictions_Proba = loaded_model.predict_proba(data_scaled)

    hour = np.sort(np.unique(data["Hours"]))
    Result = pd.DataFrame(np.column_stack((hour, Predictions, Predictions_Proba)),
                          columns=["Time of the day", "Prediction", "Probability of Flow", "Probability of Congestion"])
    Result.drop(columns="Probability of Flow", inplace=True)
    Result = Result.style.apply(add_color, subset= ["Prediction","Probability of Congestion"], axis = 1)

    #st.dataframe(Result, 1000, 1000)

    return Result

#Data Files names
data_names = {

    "(1, 2)": "1-2",
    "(1, 19)": "1-19",
    "(2, 3)": "2-3",
    "(2, 1)": "2-1",
    "(3, 4)": "3-4",
    "(3, 19)": "3-19",
    "(3, 2)": "3-2",
    "(4, 5)": "4-5",
    "(4, 3)": "4-3",
    "(5, 6)": "5-6",
    "(5, 4)": "5-4",
    "(6, 7)": "6-7",
    "(6, 15)": "6-15",
    "(6, 5)": "6-5",
    "(7, 8)": "7-8",
    "(7, 9)": "7-9",
    "(7, 6)": "7-6",
    "(8, 9)": "8-9",
    "(8, 7)": "8-7",
    "(9, 10)": "9-10",
    "(9, 8)": "9-8",
    "(9, 7)": "9-7",
    "(10, 11)": "10-11",
    "(10, 15)": "10-15",
    "(10, 9)": "10-9",
    "(11, 12)": "11-12",
    "(11, 16)": "11-16",
    "(11, 10)": "11-10",
    "(12, 13)": "12-13",
    "(12, 11)": "12-11",
    "(13, 14)": "13-14",
    "(13, 17)": "13-17",
    "(13, 12)": "13-12",
    "(14, 22)": "14-22",
    "(14, 13)": "14-13",
    "(15, 10)": "15-10",
    "(15, 16)": "15-16",
    "(15, 6)": "15-6",
    "(16, 11)": "16-11",
    "(16, 17)": "16-17",
    "(16, 21)": "16-21",
    "(16, 15)": "16-15",
    "(17, 13)": "17-13",
    "(17, 18)": "17-18",
    "(17, 16)": "17-16",
    "(18, 22)": "18-22",
    "(19, 20)": "19-20",
    "(19, 3)": "19-3",
    "(19, 1)": "19-1",
    "(20, 21)": "20-21",
    "(20, 19)": "20-19",
    "(21, 16)": "21-16",
    "(21, 20)": "21-20",
    "(22, 14)": "22-14",
    "(22, 18)": "22-18"

}

#Convert the path in to pairs of two and append it in a list.
def get_value_pairs(lst):
    pairs = []
    for i in range(len(lst) - 1):
        pairs.append((int(lst[i]), int(lst[i+1])))
    return pairs

#Iterate over the pairs of the path and get the names of the data from the dictionary to predict congestion.
def get_data_file(path, data):
  pairs = get_value_pairs(list(path))
  data_file = []
  # Iterating over the pairs
  for pair in pairs:
    data_num = data[str(pair)]
    data_file.append(data_num)

  return data_file

#Function to find the common nodes between two paths
def common(data_path_no_block, data_path_block ):
  # Identify common nodes
  common_nodes = []
  for node in data_path_no_block:
    if node in data_path_block:
      common_nodes.append(node)
  return common_nodes

#Function to find the unique nodes between two parts
def unique(data_path_no_block, common_nodes):
  # Identify unique nodes in path1
  unique_nodes = []
  for node in data_path_no_block:
    if node not in common_nodes:
      unique_nodes.append(node)
  return unique_nodes

#func to set the additional data for alternative paths
def sep_data(unique_nodes):
    # Create a separate data frame from unique nodes
    separate_df = pd.DataFrame()
    for node in unique_nodes:
        df = pd.read_excel(f"D://Final Year Project//FINAL DATA FILES//{node}.xlsx")
        separate_df = pd.concat([separate_df, df])
    # Calculate the number of rows to add to each individual data frame
    rows_to_add = int(len(separate_df) * 0.5)
    sep_data = separate_df.sample(rows_to_add)

    return sep_data



#get the data files and predict results.
def Data_Result(path, loaded_model, common_nodes, seperate_data):
    # Extracting the data according to the shortest path
    data_sets = get_data_file(path, data_names)
    selected_dataset = st.selectbox("Select The Location You Want Predictions For:", data_sets)

    if selected_dataset in common_nodes:
        data_file_path = f"D://Final Year Project//FINAL DATA FILES//{selected_dataset}.xlsx"
        Data = pd.read_excel(data_file_path)
        st.write(f"Prediction results for {selected_dataset}:")
        r = prediction_result(Data, loaded_model)

    elif selected_dataset not in common_nodes:
        data_file_path = f"D://Final Year Project//FINAL DATA FILES//{selected_dataset}.xlsx"
        Data = pd.read_excel(data_file_path)
        Data = pd.concat([Data, seperate_data], ignore_index=True)
        st.write(f"Prediction results for {selected_dataset}:")
        r = prediction_result(Data, loaded_model)
    return r

#add data set
df = pd.read_excel("D://University Final Year Project//Latitude Longitude.xlsx")
df1 = pd.read_excel("D://University Final Year Project//lat long only.xlsx")
#data = pd.read_excel("D://Final Year Project//demo.xlsx")

#Defining the scaler to scale the data
scaler = StandardScaler()

#defing the variable for width and height of the map
map_width = 1000
map_height = 700

#web Tool
st.title(""" Final Year Project: "The Development Of Artificial Intelligence Based Optimal Route Selection Tool For Peshawar Traffic Police And Rescue Service" """)

Action = st.sidebar.selectbox("Please choose", ("See selected path","Blocking Nodes Map", "Visualize the Alternative path", "Generate congestion prediction" ))

#variable to store where to block the road
a = st.sidebar.text_input("""
# Starting Node:
(Please select "Blocking Nodes Map" to visualize which is your starting node)  
""")

b = st.sidebar.text_input("""
# Destination Node:
(Please select "Blocking Nodes Map" to visualize which is your destination node)  
""")

Block = st.sidebar.text_input("""
# Blocking Node:
(Please select "Blocking Nodes Map" to see which node you want to block from 1 - 22)  
""")

st.write("# MAP: ")
try:
    path_no_block = final_astar_noblock(a,b, Map)
    path= final_astar(a,b, Block)

    data_path_no_block = get_data_file(path_no_block, data_names)
    data_path_block = get_data_file(path, data_names)

    # defining the common, unique and seperate nodes and data.
    common_nodes = common(data_path_no_block, data_path_block)
    unique_nodes = unique(data_path_no_block, common_nodes)
    seperate_data = sep_data(unique_nodes)

except:
    st.write(""" Enter the other details by using "Blocking Node Map" """)

if Action == "See selected path":
    map = st_folium(selected_path_map(df))

elif Action == "Blocking Nodes Map":
    map = st_folium(number_icon_map(df1))

elif Action == "Visualize the Alternative path":
    try:
        st.sidebar.write("# Alternative Path:")
        st.sidebar.write(path_names(path))
        map = st_folium(shortest_path_map(df1, path, Block))
        
    except:
        st.write("Enter the required nodes.")


elif Action == "Generate congestion prediction":
    map = st_folium(shortest_path_map(df1, path, Block))

else:
    map = st_folium(folium.Map(location=[34.007285, 71.560445], zoom_start=14.4))


st.write(" # Predictions: ")
if Action == "Generate congestion prediction":
    Final_result = Data_Result(path, loaded_model, common_nodes, seperate_data)
    st.dataframe(Final_result, 1000, 1000)
    st.write("0 : Traffic is flowing.  ,"
             "1: Traffic is jam (Congestion).")
    st.write("Green Color : 0% - 25% chance of Congestion")
    st.write("Yellow Color : 26% - 50% chance of Congestion")
    st.write("Orange Color : 51% - 75% chance of Congestion")
    st.write("Red Color : 76% to 100% chance of Congestion")

else:
    st.write("""
    1: Enter your Starting Node.(Take help from Blocking node map to visualize your node)
    
    2: Enter your Destination Node. (Take help from Blocking node map to visualize your node)
    
    3: Enter your Blocking Node. (Take help from Blocking node map to visualize your node)
    
    4: Than select "Generate congestion prediction".    
    """)









