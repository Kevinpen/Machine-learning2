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

plot(title="Two Player 10 Matches",
    xlabel = "Player 1 Skill",
    ylabel = "Player 2 Skill"
   )
jt10(zs)=exp.(all_games_log_likelihood(zs,games))
skillcontour!(jt10,colour=1)
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
  zs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .*exp.(params[2]) .+ params[1]
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


function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-6, num_q_samples = 10)
  params_cur = init_params
  elbo_val=neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples)
  #Generate true prior
  pzs=randn(size(init_params[1])[1],num_q_samples)
  jointp(pzs)=exp.(joint_log_density(pzs,toy_evidence)) #function for contour plot
  #initialize plot
  plot(title="Fit Toy Variational Dist",
      xlabel = "Player A Skill",
      ylabel = "Player B Skill"
     )
  display(skillcontour!(jointp,colour="red"))
  for i in 1:num_itrs
    f(params)=neg_toy_elbo(params; games = toy_evidence, num_samples = num_q_samples)
    grad_params = gradient(f, params_cur)[1]
    params_cur =  params_cur .- grad_params .* lr
    elbo_val=neg_toy_elbo(params_cur; games = toy_evidence, num_samples = num_q_samples)
    #@info "loss: $(elbo_val) "
    #U=rand(size(init_params[1])[1],num_q_samples)
    #qzs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .*exp.(params_cur[2]) .+ params_cur[1]
    #jointq(qzs)=exp.(factorized_gaussian_log_density(params_cur[1],params_cur[2],qzs) .+
    #all_games_log_likelihood(qzs,toy_evidence))
    #display(skillcontour!(jointq,colour=1))
  end
  #plot_line_equal_skill!()
  return params_cur,elbo_val
end



# Toy game
num_players_toy = 2
toy_mu = [-1.,0.8] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.2] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)
toy_evidence=two_player_toy_games(10,10)
fit=fit_toy_variational_dist(toy_params_init, toy_evidence; num_itrs=200, lr= 1e-8, num_q_samples = 10)
opt_params=fit[1]
pzs=randn(size(toy_params_init[1])[1],10)
jointp(pzs)=exp.(joint_log_density(pzs,toy_evidence)) #function for contour plot
U=rand(size(toy_params_init[1])[1],10)
qzs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .*exp.(opt_params[2]) .+ opt_params[1]
jointq(qzs)=exp.(factorized_gaussian_log_density(opt_params[1],opt_params[2],qzs) .+ all_games_log_likelihood(qzs,toy_evidence))
print("Final loss:",fit[2])
#initialize plot
plot(title="Fit Toy Variational Dist",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
display(skillcontour!(jointp,colour="red"))
display(skillcontour!(jointq,colour=1))
plot_line_equal_skill!()


file = matopen("tennis_data.mat")


## Question 4
# Load the Data
using MAT
vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")

three_player_games=vcat([1,2]',[2,1]',[1,3]',[1,3]',[3,2]')
pair_games=vcat([1,2]',[2,1]')

zs=randn(3,10)
z2=zs[1:2,:]


all_games=all_games_log_likelihood(zs,three_player_games)
pa2(z2)=exp.(log_prior(z2) .+ all_games_log_likelihood(z2,pair_games))

p3(z2)=exp.(log_prior(z2) .+all_games)
plot(title="Fit Toy Variational Dist",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
skillcontour!(pa2,colour="blue")
skillcontour!(p3,colour="red")




tennis_games[tennis_games[:,1][i]==1 for i in 1:107]
tennis_games[filter(x->x==1,tn),:]
tn=tennis_games
player1_games=[[]]
for i in 1:length(tennis_games)
  if tennis_games[i,1]==1
    append!(player1_games, tennis_games[i,:])
  end
end
p1=fill(Int[], 1, 2)

reshape([1,2],1,2)

function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  elbo_val=neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples)
  for i in 1:num_itrs
    f(params)=neg_toy_elbo(params; games = tennis_games, num_samples = num_q_samples)
    grad_params = gradient(f, params_cur)[1]
    params_cur =  params_cur .- grad_params .* lr
    elbo_val=neg_toy_elbo(params_cur; games = tennis_games, num_samples = num_q_samples)
    @info "loss: $(elbo_val) "
  end
  return params_cur, elbo_val
end

num_q_samples = 10
init_mu = randn(num_players, num_q_samples)
init_log_sigma = randn(num_players, num_q_samples)
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games,lr= 1e-10)
print("Final negative ELBO:", trained_params[2])
opt_params=trained_params[1]
means=vec(sum(opt_params[1], dims=2))
perm = sortperm(means)
print(perm[98:107])

plot(means[perm])

meanp1=mean(opt_params[1][1,:])
logsp1=mean(opt_params[2][1,:])
meanp2=mean(opt_params[1][5,:])
logsp2=mean(opt_params[2][5,:])
meanp=vcat(meanp1,meanp2)
logsp=vcat(logsp1,logsp2)
U=rand(2,10)
zs = sqrt.(-2.0 .* log.(U)) .* cos.(2*pi .* U) .* exp.(logsp) .+ meanp
jointp(zs)=exp.(factorized_gaussian_log_density(meanp,logsp,zs))
plot(title="Fit Toy Variational Dist",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
skillcontour!(jointp,colour="red")

#TODO: 10 players with highest mean skill under variational model
#hint: use sortperm

#TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
