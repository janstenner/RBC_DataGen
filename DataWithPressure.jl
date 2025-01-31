using LinearAlgebra
using Oceananigans
using Statistics
using JSON




runs = 3
collect_pressure = false

frames_to_collect = 130
start_frame = 75

num_actuators = 12

Lx = 2*pi
Lz = 2

Nx = 96
Nz = 64

Δt = 0.03
Δt_snap = 0.09
duration = 24.3

start_steps = 20

Ra = 1e4
Pr = 0.71

Re = sqrt(Ra/Pr)

ν = 1 / Re
κ = 1 / sqrt(Ra*Pr)#Re

Δb = 1

# Set the amplitude of the random perturbation (kick)
kick = 0.2

chebychev_z = false

actions = ones(12)



if chebychev_z
    chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))
    Δt = 0.00012
else
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    Δt = 0.03
end


function collate_actions_colin(actions, x, t)

    domain = Lx 

    ampl = 0.75  

    dx = 0.03  

    values = ampl.*actions
    Mean = mean(values)
    K2 = maximum([1.0, maximum(abs.(values .- Mean)) / ampl])


    segment_length = domain/num_actuators

    # determine segment of x
    x_segment = Int(floor(x / segment_length) + 1)

    if x_segment == 1
        T0 = 2 + (ampl * actions[end] - Mean)/K2
    else
        T0 = 2 + (ampl * actions[x_segment - 1] - Mean)/K2
    end

    T1 = 2 + (ampl * actions[x_segment] - Mean)/K2

    if x_segment == num_actuators
        T2 = 2 + (ampl * actions[1] - Mean)/K2
    else
        T2 = 2 + (ampl * actions[x_segment + 1] - Mean)/K2
    end

    # x position in the segment
    x_pos = x - (x_segment - 1) * segment_length

    # determine if x is in the transition regions

    if x_pos < dx

        #transition region left
        return T0+((T0-T1)/(4*dx^3)) * (x_pos - 2*dx) * (x_pos + dx)^2

    elseif x_pos >= segment_length - dx

        #transition region right
        return T1+((T1-T2)/(4*dx^3)) * (x_pos - segment_length - 2*dx) * (x_pos - segment_length + dx)^2

    else

        # middle of the segment
        return T1

    end
end

function bottom_T(x, t)
    global actions
    collate_actions_colin(actions,x,t)
end


# test plot
# xx = collect(LinRange(0,2*pi-0.0000001,1000))

# res = Float64[]

# for x in xx
#     append!(res, bottom_T(x,0))
# end

# plot(scatter(x=xx,y=res))



u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(1),
                                bottom = ValueBoundaryCondition(bottom_T))#1+Δb))



totalsteps = Int(duration/Δt_snap)





function generate_data()

    global runs
    global collect_pressure
    global frames_to_collect
    global start_frame

    if collect_pressure
        channels = 5
    else
        channels = 3
    end

    global sim_results = zeros(Float32,Nx,Nz,channels,frames_to_collect,runs)

    for n in 1:runs
        model = NonhydrostaticModel(; grid,
                    advection = UpwindBiasedFifthOrder(),
                    timestepper = :RungeKutta3,
                    tracers = (:b),
                    buoyancy = Buoyancy(model=BuoyancyTracer()),
                    closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                    boundary_conditions = (u = u_bcs, b = b_bcs,),
                    coriolis = nothing
        )

        # Set initial conditions
        uᵢ(x, z) = kick * randn()
        wᵢ(x, z) = kick * randn()
        bᵢ(x, z) =  1 + (2 - z) * Δb/2 + kick * randn()

        set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

        # Now, we create a 'simulation' to run the model for a specified length of time
        simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap)

        cur_time = 0.0

        simulation.verbose = false


        

        for i in 1:start_frame+frames_to_collect
            global simulation.stop_time = Δt_snap*i

            run!(simulation)
            cur_time += Δt_snap

            if i>start_frame
                j = i - start_frame

                sim_results[:,:,1,j,n] = model.tracers.b[1:Nx,1,1:Nz]
                sim_results[:,:,2,j,n] = model.velocities.u[1:Nx,1,1:Nz]
                sim_results[:,:,3,j,n] = model.velocities.w[1:Nx,1,1:Nz]

                if collect_pressure
                    sim_results[:,:,4,j,n] = model.pressures.pHY′[1:Nx,1,1:Nz]
                    sim_results[:,:,5,j,n] = model.pressures.pNHS[1:Nx,1,1:Nz]
                end
            end

            println(cur_time)
        end
    end



    open("rbc_data.json", "w") do f
        JSON.print(f, sim_results)
        println(f)
    end


end


generate_data()