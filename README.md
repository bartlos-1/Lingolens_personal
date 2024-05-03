## GOOGLE COLLAB LINK FOR MODEL TRAINING: 
### https://colab.research.google.com/drive/19SMEI9KyozpNurVAADf_00G2UNjW_Hv-?authuser=4

## STREAMLIT APP LINK WHERE YOU CAN TRY IT:
### https://lingoread.streamlit.app/

## RESEARCH PAPER LINK:
### https://lingopaper.tiiny.site/

## TO RUN ON YOUR OWN:
### In order to run the code the following dependencies of these exact versions are needed:

* tensorflow                    2.15.1
* imageio                       2.23.0
* opencv-python                 4.9.0.80
* numpy                         1.24.3
* keras                         2.15.0
* ffmpeg			                  7.0
* streamlit                     1.33.0

### FOR WINDOWS USERS:

#### In both google collab and in the app code:

##### In load.data() function uncomment the line:
`#file_name = path.split('\\')[-1].split('.')[0]`
#### And put in comments the line:
`file_name = path.split('/')[-1].split('.')[0]`

### TO RUN THE APP:

#### Run the following line of the code in terminal when in ‘app’ folder of the project:
`streamlit run streamlitapp.py`
