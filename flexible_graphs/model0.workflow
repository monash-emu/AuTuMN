digraph {
	graph [fontsize=16 label="Dynamic Transmission Model"]
	node [fillcolor="#CCDDFF" fontname=Helvetica shape=box style=filled]
	edge [arrowhead=open fontname=Courier fontsize=10 style=dotted]
		susceptible_fully_nocomorbs
		susceptible_vac_nocomorbs
		susceptible_treated_nocomorbs
		latent_early_ds_nocomorbs
		latent_early_mdr_nocomorbs
		latent_late_ds_nocomorbs
		latent_late_mdr_nocomorbs
		active_smearpos_ds_nocomorbs
		active_smearneg_ds_nocomorbs
		active_extrapul_ds_nocomorbs
		active_smearpos_mdr_nocomorbs
		active_smearneg_mdr_nocomorbs
		active_extrapul_mdr_nocomorbs
		detect_smearpos_ds_nocomorbs
		detect_smearneg_ds_nocomorbs
		detect_extrapul_ds_nocomorbs
		detect_smearpos_mdr_nocomorbs
		detect_smearneg_mdr_nocomorbs
		detect_extrapul_mdr_nocomorbs
		missed_smearpos_ds_nocomorbs
		missed_smearneg_ds_nocomorbs
		missed_extrapul_ds_nocomorbs
		missed_smearpos_mdr_nocomorbs
		missed_smearneg_mdr_nocomorbs
		missed_extrapul_mdr_nocomorbs
		treatment_infect_smearpos_ds_nocomorbs
		treatment_infect_smearneg_ds_nocomorbs
		treatment_infect_extrapul_ds_nocomorbs
		treatment_infect_smearpos_mdr_nocomorbs
		treatment_infect_smearneg_mdr_nocomorbs
		treatment_infect_extrapul_mdr_nocomorbs
		treatment_noninfect_smearpos_ds_nocomorbs
		treatment_noninfect_smearneg_ds_nocomorbs
		treatment_noninfect_extrapul_ds_nocomorbs
		treatment_noninfect_smearpos_mdr_nocomorbs
		treatment_noninfect_smearneg_mdr_nocomorbs
		treatment_noninfect_extrapul_mdr_nocomorbs
		tb_death
			susceptible_fully_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_ds]
			susceptible_vac_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_treated_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			latent_late_ds_nocomorbs -> latent_early_ds_nocomorbs [label=rate_force_weak_ds]
			susceptible_fully_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_mdr]
			susceptible_vac_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			susceptible_treated_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			latent_late_mdr_nocomorbs -> latent_early_mdr_nocomorbs [label=rate_force_weak_mdr]
			active_smearpos_ds_nocomorbs -> detect_smearpos_ds_nocomorbs [label=program_rate_detect]
			active_smearpos_ds_nocomorbs -> missed_smearpos_ds_nocomorbs [label=program_rate_missed]
			active_smearneg_ds_nocomorbs -> detect_smearneg_ds_nocomorbs [label=program_rate_detect]
			active_smearneg_ds_nocomorbs -> missed_smearneg_ds_nocomorbs [label=program_rate_missed]
			active_extrapul_ds_nocomorbs -> detect_extrapul_ds_nocomorbs [label=program_rate_detect]
			active_extrapul_ds_nocomorbs -> missed_extrapul_ds_nocomorbs [label=program_rate_missed]
			active_smearpos_mdr_nocomorbs -> detect_smearpos_mdr_nocomorbs [label=program_rate_detect]
			active_smearpos_mdr_nocomorbs -> missed_smearpos_mdr_nocomorbs [label=program_rate_missed]
			active_smearneg_mdr_nocomorbs -> detect_smearneg_mdr_nocomorbs [label=program_rate_detect]
			active_smearneg_mdr_nocomorbs -> missed_smearneg_mdr_nocomorbs [label=program_rate_missed]
			active_extrapul_mdr_nocomorbs -> detect_extrapul_mdr_nocomorbs [label=program_rate_detect]
			active_extrapul_mdr_nocomorbs -> missed_extrapul_mdr_nocomorbs [label=program_rate_missed]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=program_rate_default_infect_amplify]
			treatment_noninfect_smearpos_ds_nocomorbs -> active_smearpos_mdr_nocomorbs [label=program_rate_default_noninfect_amplify]
			treatment_infect_smearneg_ds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=program_rate_default_infect_amplify]
			treatment_noninfect_smearneg_ds_nocomorbs -> active_smearneg_mdr_nocomorbs [label=program_rate_default_noninfect_amplify]
			treatment_infect_extrapul_ds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=program_rate_default_infect_amplify]
			treatment_noninfect_extrapul_ds_nocomorbs -> active_extrapul_mdr_nocomorbs [label=program_rate_default_noninfect_amplify]
			latent_early_ds_nocomorbs -> latent_late_ds_nocomorbs [label=2.3]
			latent_early_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.13]
			latent_late_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.0029]
			active_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.07]
			detect_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.07]
			missed_smearpos_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.07]
			latent_early_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.19]
			latent_late_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.0040]
			active_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			detect_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			missed_smearneg_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			latent_early_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.0056]
			latent_late_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.000120]
			active_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			detect_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			missed_extrapul_ds_nocomorbs -> latent_late_ds_nocomorbs [label=0.20]
			latent_early_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=2.3]
			latent_early_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.13]
			latent_late_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.0029]
			active_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.07]
			detect_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.07]
			missed_smearpos_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.07]
			latent_early_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.19]
			latent_late_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.0040]
			active_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			missed_smearneg_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			latent_early_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.0056]
			latent_late_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.000120]
			active_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			missed_extrapul_mdr_nocomorbs -> latent_late_mdr_nocomorbs [label=0.20]
			detect_smearpos_ds_nocomorbs -> treatment_infect_smearpos_ds_nocomorbs [label=25.6]
			missed_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=4.0]
			detect_smearneg_ds_nocomorbs -> treatment_infect_smearneg_ds_nocomorbs [label=25.6]
			missed_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=4.0]
			detect_extrapul_ds_nocomorbs -> treatment_infect_extrapul_ds_nocomorbs [label=25.6]
			missed_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=4.0]
			detect_smearpos_mdr_nocomorbs -> treatment_infect_smearpos_mdr_nocomorbs [label=25.6]
			missed_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=4.0]
			detect_smearneg_mdr_nocomorbs -> treatment_infect_smearneg_mdr_nocomorbs [label=25.6]
			missed_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=4.0]
			detect_extrapul_mdr_nocomorbs -> treatment_infect_extrapul_mdr_nocomorbs [label=25.6]
			missed_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=4.0]
			treatment_infect_smearpos_ds_nocomorbs -> treatment_noninfect_smearpos_ds_nocomorbs [label=28.1]
			treatment_noninfect_smearpos_ds_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.14]
			treatment_noninfect_smearpos_ds_nocomorbs -> active_smearpos_ds_nocomorbs [label=0.13]
			treatment_infect_smearpos_mdr_nocomorbs -> treatment_noninfect_smearpos_mdr_nocomorbs [label=28.1]
			treatment_noninfect_smearpos_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.15]
			treatment_noninfect_smearpos_mdr_nocomorbs -> active_smearpos_mdr_nocomorbs [label=0.14]
			treatment_infect_smearneg_ds_nocomorbs -> treatment_noninfect_smearneg_ds_nocomorbs [label=28.1]
			treatment_noninfect_smearneg_ds_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.14]
			treatment_noninfect_smearneg_ds_nocomorbs -> active_smearneg_ds_nocomorbs [label=0.13]
			treatment_infect_smearneg_mdr_nocomorbs -> treatment_noninfect_smearneg_mdr_nocomorbs [label=28.1]
			treatment_noninfect_smearneg_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.15]
			treatment_noninfect_smearneg_mdr_nocomorbs -> active_smearneg_mdr_nocomorbs [label=0.14]
			treatment_infect_extrapul_ds_nocomorbs -> treatment_noninfect_extrapul_ds_nocomorbs [label=28.1]
			treatment_noninfect_extrapul_ds_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.14]
			treatment_noninfect_extrapul_ds_nocomorbs -> active_extrapul_ds_nocomorbs [label=0.13]
			treatment_infect_extrapul_mdr_nocomorbs -> treatment_noninfect_extrapul_mdr_nocomorbs [label=28.1]
			treatment_noninfect_extrapul_mdr_nocomorbs -> susceptible_treated_nocomorbs [label=1.8]
			treatment_infect_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.15]
			treatment_noninfect_extrapul_mdr_nocomorbs -> active_extrapul_mdr_nocomorbs [label=0.14]
			active_smearpos_ds_nocomorbs -> tb_death [label=0.18]
			detect_smearpos_ds_nocomorbs -> tb_death [label=0.18]
			missed_smearpos_ds_nocomorbs -> tb_death [label=0.18]
			active_smearneg_ds_nocomorbs -> tb_death [label=0.0466]
			detect_smearneg_ds_nocomorbs -> tb_death [label=0.0466]
			missed_smearneg_ds_nocomorbs -> tb_death [label=0.0466]
			active_extrapul_ds_nocomorbs -> tb_death [label=0.0466]
			detect_extrapul_ds_nocomorbs -> tb_death [label=0.0466]
			missed_extrapul_ds_nocomorbs -> tb_death [label=0.0466]
			active_smearpos_mdr_nocomorbs -> tb_death [label=0.18]
			detect_smearpos_mdr_nocomorbs -> tb_death [label=0.18]
			missed_smearpos_mdr_nocomorbs -> tb_death [label=0.18]
			active_smearneg_mdr_nocomorbs -> tb_death [label=0.0466]
			detect_smearneg_mdr_nocomorbs -> tb_death [label=0.0466]
			missed_smearneg_mdr_nocomorbs -> tb_death [label=0.0466]
			active_extrapul_mdr_nocomorbs -> tb_death [label=0.0466]
			detect_extrapul_mdr_nocomorbs -> tb_death [label=0.0466]
			missed_extrapul_mdr_nocomorbs -> tb_death [label=0.0466]
			treatment_infect_smearpos_ds_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_smearpos_ds_nocomorbs -> tb_death [label=0.0473]
			treatment_infect_smearpos_mdr_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_smearpos_mdr_nocomorbs -> tb_death [label=0.0473]
			treatment_infect_smearneg_ds_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_smearneg_ds_nocomorbs -> tb_death [label=0.0473]
			treatment_infect_smearneg_mdr_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_smearneg_mdr_nocomorbs -> tb_death [label=0.0473]
			treatment_infect_extrapul_ds_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_extrapul_ds_nocomorbs -> tb_death [label=0.0473]
			treatment_infect_extrapul_mdr_nocomorbs -> tb_death [label=0.0479]
			treatment_noninfect_extrapul_mdr_nocomorbs -> tb_death [label=0.0473]
}