using Revise # lets you change A2funcs without restarting julia!
includet("A2_src.jl")
using Plots
using Statistics: mean
using Zygote 
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable 
using .A2funcs: factorized_gaussian_log_density 
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!

function log_prior(zs)
  return  #TODO
end

function logp_a_beats_b(za,zb)
  return #TODO
end


function all_games_log_likelihood(zs,games)
  zs_a = #TODO
  zs_b =  #TODO
  likelihoods =  #TODO
  return  #TODO
end

function joint_log_density(zs,games)
  return #TODO
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
plot(title="Example Gaussian Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
skillcontour!(example_gaussian; label="example gaussian")
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))


# TODO: plot prior contours

# TODO: plot likelihood contours

# TODO: plot joint contours with player A winning 1 game

# TODO: plot joint contours with player A winning 10 games

#TODO: plot joint contours with player A winning 10 games and player B winning 10 games

function elbo(params,logp,num_samples)
  samples = #TODO
  logp_estimate = #TODO
  logq_estimate = #TODO
  return #TODO: should return scalar (hint: average over batch)
end

# Conveinence function for taking gradients 
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  # TODO: Write a function that takes parameters for q, 
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


# Toy game
num_players_toy = 2 
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)



function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params 
  for i in 1:num_itrs
    grad_params = #TODO: gradients of variational objective with respect to parameters
    params_cur =  #TODO: update paramters with lr-sized step in descending gradient
    @info #TODO: report the current elbbo during training
    # TODO: plot true posterior in red and variational in blue
    # hint: call 'display' on final plot to make it display during training
    plot();
    #TODO: skillcontour!(...,colour=:red) plot likelihood contours for target posterior
    # plot_line_equal_skill()
    #TODO: display(skillcontour!(..., colour=:blue)) plot likelihood contours for variational posterior
  end
  return params_cur
end

#TODO: fit q with SVI observing player A winning 1 game
#TODO: save final posterior plots

#TODO: fit q with SVI observing player A winning 10 games
#TODO: save final posterior plots

#TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
#TODO: save final posterior plots

## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params 
  for i in 1:num_itrs
    grad_params = #TODO: gradients of variational objective wrt params
    params_cur = #TODO: update parmaeters wite lr-sized steps in desending gradient direction
    @info #TODO: report objective value with current parameters
  end
  return params_cur
end

# TODO: Initialize variational family
init_mu = #random initialziation
init_log_sigma = # random initialziation
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)


#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm

#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
