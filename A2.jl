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
  factorized_gaussian_log_density(0,0,zs)
end

function logp_a_beats_b(za,zb)
  return log.(1 ./exp.(log1pexp.(-(za .- zb))))
end


function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:,1],:]
  zs_b = zs[games[:,2],:]
  likelihoods = sum(logp_a_beats_b(zs_a,zs_b),dims=1)
  return  likelihoods
end


function joint_log_density(zs,games)
  return log_prior(zs) .+ all_games_log_likelihood(zs,games)
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
skillcontour!(example_gaussian,colour=2)
plot_line_equal_skill!()
savefig(joinpath("plots","example_gaussian.pdf"))


# TODO: plot prior contours

jointPrior(zs) = exp.(log_prior(zs))
plot(title="Two Player Joint Prior Contour Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(jointPrior)
plot_line_equal_skill!()


# TODO: plot likelihood contours
likelihood(zs)=exp.(logp_a_beats_b(zs[1],zs[2]))
plot(title="Two Player likelihood Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(likelihood,colour=3)
plot_line_equal_skill!()

# TODO: plot joint contours with player A winning 1 game
games=two_player_toy_games(5, 0)
zs = randn(2,15)*2
jt(zs)=exp(all_games_log_likelihood(zs,games))
jtd(zs)=exp(joint_log_density(zs,games))
plot(title="Two Player Joint Posterior Plot",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(jt,colour=1)
skillcontour!(jtd,colour=1)
plot_line_equal_skill!()


# TODO: plot joint contours with player A winning 10 games
games=two_player_toy_games(1, 0)
skillcontour!(jt10,colour=1)
plot(title="Two Player 10 Matches",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
jt10(zs)=exp.(all_games_log_likelihood(zs,games))
plot_line_equal_skill!()


#TODO: plot joint contours with player A winning 10 games and player B winning 10 games
games=two_player_toy_games(10, 10)
jt20(zs)=exp.(all_games_log_likelihood(zs,games))
plot(title="Two Player 20 Matches",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
skillcontour!(jt20,colour=1)
plot_line_equal_skill!()


function elbo(params,logp,num_samples)
  #Generate random samples from uniform distribution
  U=rand(size(params[1])[1],num_samples)
  #Reparametrization to genearte Gaussian of desired parameter
  zs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .*params[2] .+ params[1]
  log_z=factorized_gaussian_log_density(0,0,zs)
  logp_estimate = logp
  log_data=logp .- log_z #Separate data from logp
  #estimate $q_{Φ}(z|data)$
  logq_estimate = factorized_gaussian_log_density(params[1],params[2],zs) .+ log_data
  return sum(logp_estimate .- logq_estimate) ./ num_samples
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  zs=randn(size(params[1])[1],num_samples)
  logp = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.2] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)
toy_evidence=two_player_toy_games(3,2)

function joint_log_density(zs,games)
  return log_prior(zs) .+ all_games_log_likelihood(zs,games)
end

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  #Generate true prior
  pzs=randn(size(init_params[1])[1],num_q_samples)
  jointp(pzs)=exp.(joint_log_density(pzs,toy_evidence)) #function for contour plot

  #initialize plot
  plot(title="Fit Toy Variational Dist",
      xlabel = "Player A Skill",
      ylabel = "Player B Skill"
     )
  for i in 1:num_itrs
    elbo=neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples)
    f(params)=neg_toy_elbo(params; games = toy_evidence, num_samples = num_q_samples)
    grad_params = gradient(f, params_cur)[1]
    params_cur =  params_cur .- grad_params .* lr
    @info "loss: $(elbo), params_cur:$(params_cur) "
    #TODO: skillcontour!(...,colour=:red) plot likelihood contours for target posterior
    display(skillcontour!(jointp,colour="red"))
    #TODO: display(skillcontour!(..., colour=:blue)) plot likelihood contours for variational posterior
    U=rand(size(init_params[1])[1],num_q_samples)
    qzs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .*params_cur[2] .+ params_cur[1]
    jointq(qzs)=exp.(factorized_gaussian_log_density(params_cur[1],params_cur[2],qzs) .+
    all_games_log_likelihood(zs,toy_evidence))
    display(skillcontour!(jointq,colour=1))
  end
  plot_line_equal_skill!()
  return params_cur
end

fit_toy_variational_dist(toy_params_init, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
params_ret=([-0.30823121619653104, 0.3631615384805915], [0.925795140030016, 0.7914438293357612])
xs=randn(1,2)
zs=factorized_gaussian_log_density(params_ret[1],params_ret[2],xs)
postq(zs)=exp.(joint_log_density(zs,toy_evidence))
skillcontour!(postq,colour=2)

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
