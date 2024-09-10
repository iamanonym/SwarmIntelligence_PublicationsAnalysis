# SwarmIntelligence_PublicationsAnalysis
"SwarmIntelligence_PublicationsAnalysis" is a software for local performing of visualizing tasks on publication activity data that was collected on the theme of Swarm Intelligence. 

"SwarmIntelligence_PublicationsAnalysis" is specifically designed for the collaborative tasks of Mariana College and NMSTU researchers in the field of publication activity, firstly for the current task of Swarm Intelligence publication analysis. 

# Preparing data
Visualizing requires previously prepared data in the specific format. The main/static directory contains files "Data_2024.csv" and "Tutors_list_Scopus.csv". These are templates for preparing publication activity data and here you should put data for visualization. The first one requires data on publications and the second one requires data on researchers.

# Working with application
After launching the application the local API-service that serves commands /countries/{width}/{height} and /keywords/{width}/{height} and sends SVG-files of specified size with publication collaboration scheme in the response.
