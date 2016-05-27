digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully
		susceptible_vac
		susceptible_treated
		latent_early
		latent_late
		active
		detect
		missed
		treatment_infect
		treatment_noninfect
		tb_death
			susceptible_fully -> latent_early [label=rate]
			susceptible_vac -> latent_early [label=rate]
			susceptible_treated -> latent_early [label=rate]
			latent_late -> latent_early [label=rate]
			active -> missed [label=prog]
			active -> detect [label=prog]
			treatment_infect -> treatment_noninfect [label=prog]
			treatment_noninfect -> susceptible_treated [label=prog]
			treatment_infect -> active [label=prog]
			treatment_noninfect -> active [label=prog]
			latent_early -> latent_late [label=2.2]
			latent_early -> active [label=0.30]
			latent_late -> active [label=0.0070]
			active -> latent_late [label=0.20]
			missed -> latent_late [label=0.20]
			detect -> latent_late [label=0.20]
			missed -> active [label=4.0]
			detect -> treatment_infect [label=26.0]
			active -> tb_death [label=0.13]
			missed -> tb_death [label=0.13]
			detect -> tb_death [label=0.13]
			treatment_infect -> tb_death [label=prog]
			treatment_noninfect -> tb_death [label=prog]
}