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
			susceptible_fully -> latent_early [label=rate_force]
			susceptible_vac -> latent_early [label=rate_force_weak]
			susceptible_treated -> latent_early [label=rate_force_weak]
			latent_late -> latent_early [label=rate_force_weak]
			active -> detect [label=program_rate_detect]
			active -> missed [label=program_rate_missed]
			latent_early -> latent_late [label=2.2]
			latent_early -> active [label=0.30]
			latent_late -> active [label=0.0070]
			active -> latent_late [label=0.20]
			missed -> latent_late [label=0.20]
			detect -> latent_late [label=0.20]
			detect -> treatment_infect [label=26.0]
			missed -> active [label=4.0]
			treatment_infect -> treatment_noninfect [label=28.4]
			treatment_noninfect -> susceptible_treated [label=2.0]
			treatment_infect -> active [label=0.16]
			treatment_noninfect -> active [label=0.15]
			active -> tb_death [label=0.13]
			missed -> tb_death [label=0.13]
			detect -> tb_death [label=0.13]
			treatment_infect -> tb_death [label=0.05]
			treatment_noninfect -> tb_death [label=0.0500]
}