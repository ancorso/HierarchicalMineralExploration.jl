"""
Describes a Gaussian Process model of the background grade
"""
struct GradeBackground
    μ
    kernel
end

"""
Defines the distributions used to sample geochemical domains. 
"""
@with_kw struct GeochemicalDomainDistribution
    N = 32
    cx = Distributions.Normal(N/2.0,  N/2.0)
    cy = Distributions.Normal(N/2.0,  N/2.0)
    r1 = Distributions.Normal(5, 2.5)
    r2 = Distributions.Normal(5, 2.5)
    r3 = Distributions.Normal(5, 2.5)
    r4 = Distributions.Normal(5, 2.5)
    r5 = Distributions.Normal(5, 2.5)
    r6 = Distributions.Normal(5, 2.5)
    r7 = Distributions.Normal(5, 2.5)
    r8 = Distributions.Normal(5, 2.5)
    r9 = Distributions.Normal(5, 2.5)
    r10 = Distributions.Normal(5, 2.5)
    μ           # Mean of the GP within this domain
    kernel      # Kernel of the GP within this domain
end

"""
Draws a sample of the center, points and angle that make up the blob shape
"""
Base.rand(rng::AbstractRNG, b::GeochemicalDomainDistribution) =
    (rand(rng, b.center), [rand(rng, p) for p in b.points], rand(rng, b.angle))

# """
# Samples a geochemical domain (blob shape) for the provided parameters. 
# """
# function draw_geochemical_domain(
#     N, center, pts, angle; σ=N / 25, grayscale=0.4, threshold=0.17
# )
#     pts = polysortbyangle([Luxor.Point(pt...) for pt in pts])
#     mat = @imagematrix begin
#         gsave()
#         background("black")
#         Luxor.origin(Luxor.Point(0, 0))
#         pos = Luxor.Point(center...)
#         Luxor.translate(pos)
#         Luxor.rotate(angle)
#         sethue(grayscale, grayscale, grayscale)
#         drawbezierpath(makebezierpath(pts), :fill)
#         grestore()
#     end N N

#     blurred = imfilter(mat, ImageFiltering.Kernel.gaussian(σ))

#     return Float64.(Gray.(blurred) .> threshold)
# end
function draw_geochemical_domain(N, center, rs)
    Npts = length(rs)
    thetas = range(0, 359, Npts)
    pts = Vector{Tuple{Float64, Float64}}(undef, Npts)
    for i=1:Npts
        pts[i] = center .+ (rs[i]*cosd(thetas[i]), rs[i]*sind(thetas[i]))
    end
    v = PolyArea(Ring(pts...))

    # Fill the domain with values that are in the polyarea
    domain = zeros(N, N)
    for i in 1:N, j in 1:N
        if Meshes.Point(i, j) in v
            domain[i, j] = 1.0
        end
    end

    return domain
end

"""
Defines the background GP distribution of thickness. Note the kernel is re-used inside the graben. 
"""
struct ThicknessBackground
    μ
    kernel
end

"""
Defines the distributions used to sample a Graben shape. 
"""
@with_kw struct GrabenDistribution
    N = 32
    left_top = Distributions.Normal(N/2.0, N/2.0)
    left_width = Distributions.Normal(N/4.0, N/4.0)
    right_top = Distributions.Normal(N/2.0, N/2.0)
    right_width = Distributions.Normal(N/4.0, N/4.0)
    μ           # Mean of the GP within this domain
end

"""
Draws a sample of the left top, left width, right top and right width that make up the graben shape
"""
Base.rand(rng::AbstractRNG, b::GrabenDistribution) = (
    rand(rng, b.left_top),
    rand(rng, b.left_width),
    rand(rng, b.right_top),
    rand(rng, b.right_width),
)

"""
Sample a Graben domain with the provided parameters.
"""
function draw_graben(N, ltop, lwidth, rtop, rwidth)
    # Compute the polygon representing the Graben
    v = VPolygon([[1.0, ltop], [1.0, ltop - lwidth], [N, rtop - rwidth], [N, rtop]])

    # Fill the domain with values that are in the rectangle
    domain = zeros(N, N)
    for i in 1:N, j in 1:N
        if [i, j] in v
            domain[i, j] = 1.0
        end
    end

    return domain
end
