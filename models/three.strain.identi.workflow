digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully
		susceptible_vac
		susceptible_treated
		latent_early
		latent_late
		active_smearpos
		active_smearneg
		active_extrapul
		detect_smearpos
		detect_smearneg
		detect_extrapul
		missed_smearpos
		missed_smearneg
		missed_extrapul
		treatment_infect_smearpos
		treatment_infect_smearneg
		treatment_infect_extrapul
		treatment_noninfect_smearpos
		treatment_noninfect_smearneg
		treatment_noninfect_extrapul
		tb_death
			susceptible_fully -> latent_early [label=rate_force]
			susceptible_vac -> latent_early [label=rate_force_weak]
			susceptible_treated -> latent_early [label=rate_force_weak]
			latent_late -> latent_early [label=rate_force_weak]
			latent_early -> latent_late [label=2.28880287212]
			latent_early -> active_smearpos [label=0.19531977707]
			latent_late -> active_smearpos [label=0.000342410733387]
			active_smearpos -> latent_late [label=0.102901131842]
			active_smearpos -> detect_smearpos [label=2.9323181296]
			active_smearpos -> missed_smearpos [label=0.474087159231]
			detect_smearpos -> treatment_infect_smearpos [label=25.6344915432]
			missed_smearpos -> active_smearpos [label=0.272326649365]
			missed_smearpos -> latent_late [label=0.102901131842]
			treatment_infect_smearpos -> treatment_noninfect_smearpos [label=27.9559777848]
			treatment_infect_smearpos -> active_smearpos [label=0.182996874574]
			treatment_noninfect_smearpos -> active_smearpos [label=0.174396406083]
			treatment_noninfect_smearpos -> susceptible_treated [label=1.68044664874]
			latent_early -> active_smearneg [label=0.0651065923565]
			latent_late -> active_smearneg [label=0.000114136911129]
			active_smearneg -> latent_late [label=0.29539304242]
			active_smearneg -> detect_smearneg [label=2.9323181296]
			active_smearneg -> missed_smearneg [label=0.474087159231]
			detect_smearneg -> treatment_infect_smearneg [label=25.6344915432]
			missed_smearneg -> active_smearneg [label=0.272326649365]
			missed_smearneg -> latent_late [label=0.29539304242]
			treatment_infect_smearneg -> treatment_noninfect_smearneg [label=27.9559777848]
			treatment_infect_smearneg -> active_smearneg [label=0.182996874574]
			treatment_noninfect_smearneg -> active_smearneg [label=0.174396406083]
			treatment_noninfect_smearneg -> susceptible_treated [label=1.68044664874]
			latent_early -> active_extrapul [label=0.0651065923565]
			latent_late -> active_extrapul [label=0.000114136911129]
			active_extrapul -> latent_late [label=0.29539304242]
			active_extrapul -> detect_extrapul [label=2.9323181296]
			active_extrapul -> missed_extrapul [label=0.474087159231]
			detect_extrapul -> treatment_infect_extrapul [label=25.6344915432]
			missed_extrapul -> active_extrapul [label=0.272326649365]
			missed_extrapul -> latent_late [label=0.29539304242]
			treatment_infect_extrapul -> treatment_noninfect_extrapul [label=27.9559777848]
			treatment_infect_extrapul -> active_extrapul [label=0.182996874574]
			treatment_noninfect_extrapul -> active_extrapul [label=0.174396406083]
			treatment_noninfect_extrapul -> susceptible_treated [label=1.68044664874]
			active_smearpos -> tb_death [label=0.260201067311]
			detect_smearpos -> tb_death [label=0.260201067311]
			treatment_infect_smearpos -> tb_death [label=0.182996874574]
			treatment_noninfect_smearpos -> tb_death [label=0.174396406083]
			active_smearneg -> tb_death [label=0.067709156734]
			detect_smearneg -> tb_death [label=0.067709156734]
			treatment_infect_smearneg -> tb_death [label=0.182996874574]
			treatment_noninfect_smearneg -> tb_death [label=0.174396406083]
			active_extrapul -> tb_death [label=0.067709156734]
			detect_extrapul -> tb_death [label=0.067709156734]
			treatment_infect_extrapul -> tb_death [label=0.182996874574]
			treatment_noninfect_extrapul -> tb_death [label=0.174396406083]
}