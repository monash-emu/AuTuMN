digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible
		latent_early
		latent_late
		active
		treatment_infect
		treatment_noninfect
		tb_death
			susceptible -> latent_early [label=rate_force]
			latent_early -> active [label=0.20]
			latent_early -> latent_late [label=1.8]
			latent_late -> active [label=0.0010]
			active -> latent_late [label=0.20]
			active -> treatment_infect [label=1.0]
			treatment_infect -> treatment_noninfect [label=1.8]
			treatment_infect -> active [label=0.10]
			treatment_noninfect -> susceptible [label=1.8]
			treatment_noninfect -> active [label=0.10]
			active -> tb_death [label=0.13]
			treatment_infect -> tb_death [label=0.10]
			treatment_noninfect -> tb_death [label=0.10]
}