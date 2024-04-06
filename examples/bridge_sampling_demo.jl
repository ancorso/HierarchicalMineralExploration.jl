
using Turing
using StatsPlots

@model function H1(y)
    s ~ Normal(1, 1)
    y ~ Normal(s, 1)
end

@model function H2(y)
    s ~ Normal(-1, 1)
    y ~ Normal(s, 1)
end

obs = 3.0

model1 = H1(obs)
model2 = H2(obs)

chain1 = sample(model1, NUTS(0.65), 1000, discard_initial=500)
chain2 = sample(model2, NUTS(0.65), 1000, discard_initial=500)

plot(chain1)
plot(chain2)

# For H1:
g1s = chain1[:, :lp, 1]
g2s = logpdf.(Normal(1,1), chain1[:s])[:]
gbs = (g1s .+ g2s)/2
mean(exp.(gbs) ./ exp.(g2s)) / mean(exp.(gbs) ./ exp.(g1s))

# For H2:
g1s = chain2[:, :lp, 1]
g2s = logpdf.(Normal(-1,1), chain2[:s])[:]
gbs = (g1s .+ g2s)/2
mean(exp.(gbs) ./ exp.(g2s)) / mean(exp.(gbs) ./ exp.(g1s))

# For a nulll hypothesis:
pdf(Normal(0,3), obs)