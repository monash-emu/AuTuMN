from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

import plotly.graph_objects as go


def plot_multic_relative_outputs(output_dfs_dict: dict[str, pd.DataFrame], req_outputs=["cases_averted_relative", "deaths_averted_relative", "delta_hospital_peak_relative"]):
    n_subplots = len(req_outputs)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(25, n_subplots*6))

    this_iso3_list = list(output_dfs_dict.keys())
    n_countries = len(this_iso3_list)
    ylab_lookup = {
        "cases_averted_relative": "% infections averted by school closure", 
        "deaths_averted_relative": "% deaths averted by school closure",
        "delta_hospital_peak_relative": "Relative reduction in peak hospital occupancy (%)"
    }

    box_width = .4
    med_color = 'white'
    box_colors= ['black', 'purple', 'firebrick']
    
    for i_output, output in enumerate(req_outputs):
        box_color = box_colors[i_output]
        axis = axes[i_output]

        mean_values = [output_dfs_dict[iso3][output].loc[0.5] for iso3 in this_iso3_list]
        sorted_iso3_list = [iso3 for _ , iso3 in sorted(zip(mean_values, this_iso3_list))]
        
        y_max_abs = 0.
        for i_iso3, iso3 in enumerate(sorted_iso3_list):
            x = i_iso3 + 1

            data = - 100. * output_dfs_dict[iso3][output] # use %. And use "-" so positive nbs indicate positive effect of closures

            # median
            axis.hlines(y=data.loc[0.5], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=2., color=med_color, zorder=3)    
            
            # IQR
            q_75 = data.loc[0.75]
            q_25 = data.loc[0.25]
            rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
            axis.add_patch(rect)

            # 95% CI
            q_025 = data.loc[0.025]
            q_975 = data.loc[0.975]
            axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=1.5, color=box_color, zorder=1)

            y_max_abs = max(abs(q_975), y_max_abs)
            y_max_abs = max(abs(q_025), y_max_abs)

            axis.set_xlim((0, n_countries + 1))
            axis.set_ylim(-1.2*y_max_abs, 1.2*y_max_abs)

        axis.set_xticks(ticks=range(1, n_countries + 1), labels=sorted_iso3_list, rotation=90, fontsize=13)

        y_label = ylab_lookup[output]
        axis.set_ylabel(y_label, fontsize=13)

        # add coloured backgorund patches
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim() 
        rect_up = Rectangle(xy=(xmin, 0.), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="white")  #"honeydew")
        axis.add_patch(rect_up)
        rect_low = Rectangle(xy=(xmin, ymin), width=xmax - xmin, height=(ymax - ymin)/2., zorder=-1, facecolor="gainsboro", alpha=.5)  #"mistyrose")
        axis.add_patch(rect_low)

        axis.text(n_countries * .75, ymax / 2., s="Positive effect of\nschool closures", fontsize=13)
        axis.text(n_countries * .25, ymin / 2., s="Negative effect of\nschool closures", fontsize=13)   

    plt.tight_layout()

    return fig


def plot_relative_map(output_dfs_dict: dict[str, pd.DataFrame], req_output="delta_hospital_peak_relative"):
   this_iso3_list = list(output_dfs_dict.keys())
   values = [- 100 * output_dfs_dict[iso3][req_output].loc[0.5] for iso3 in this_iso3_list]
   data_df = pd.DataFrame.from_dict({"iso3": this_iso3_list, "values": values})
   
   fig = go.Figure(
      data=go.Choropleth(
         locations=data_df["iso3"], 
         z=data_df["values"],
         colorscale= [[0, 'lightblue'],
                  #   [0.5, 'darkgrey'],
                    [1, 'blue']],  # "Plotly3",
         marker_line_color='darkgrey',
         marker_line_width=0.5,
         colorbar_title="Relative reduction in<br>peak hospital occupancy",
         colorbar_ticksuffix="%",
      )  
   )
   fig.update_layout(
      geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'       #'natural earth'   'equirectangular'
         ),
      margin={"r":0,"t":0,"l":0,"b":0},
      autosize=False,
      height=500,
      width=1200,
   )
   return fig