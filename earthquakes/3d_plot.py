from features import *
from import_csv import *
from preprocessing import *
import plotly.express as px


path = 'earthquakes/data/earthquakes_turkey.csv'
df = import_csv(path)

df = cluster_earthquakes(df, 10)

df = add_inter_event_duration(df)

df = df[df['inter_event_duration'] <= 870000]


def create_3d_scatter_plot(data):
    fig = px.scatter_3d(data, x='depth', y='magnitude', z='inter_event_duration',
                        color='cluster_id', opacity=0.7, title='Earthquake Clusters')

    fig.update_layout(scene=dict(
        xaxis_title='Depth',
        yaxis_title='Magnitude',
        zaxis_title='Inter-event Duration'
    ))

    fig.show()

create_3d_scatter_plot(df)