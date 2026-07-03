import numpy as numpy
from noise import pnoise3

# returns a value between -1 and 1 as the fractal brownian noise 
# representing the signed local density of the noise field at the exact point. It is interepreted as 
# what the odds are that rock is air.  
#   -> high value => carve 
#   -> low values => leave solid 
#
# By using fractal brownian motion we create a multiscale pattern where large caverns (octave 1) are 
# overlaid with medium tunnels (octave 2) and wall/roof roughness (octave 3+).
# Final normalisation ensures the threshold comparison is scale-invariant 
# as octives change (hence using a fractal ).
#
# @params : 
#   * (x,y,z) = current coordinates
#   * seed = offsets the noise field. 
#            used by pnoise3 as a base arg to permute the permutation table
#            [permutation table 0->255 mapping input coordinate to pseudo-random gradient, 
#            read more about Perlin's original table for an ideea ]
#   * octaves = No. layers of noise summed together
#               Each octave doubles the frequency and muls the amplitude by the persistance
#   * persistance = Rate amplitude decays per octave.  
#                   Each amplitude is * by persistance
#   * lacunarity = Rate frequency increases per wave.
#                  Mulled by each octives freqwuency 
#   * scale = base noise frequency controlling the wavelngth of the first octave
#
#   All current values were selected by AI, as such should be taken with a grain of salt.
#   Experiment with changing the values to generate different systems 
#   NOTE: 
#   octaves controls the levels of detail i.e large/ caverns and tunnel sizes
#   persistance means each octave contributes that fraction as the previous
#   scale : 0.04 means the noise function sees coordinates divided by 1/0.04 = 25 ?? (0.01 = huge cathedral caverns)


def fractal_noise(x, y, z, seed, octaves = 4, persistance = 0.5, lacunarity = 2.0, scale = 0.04):
    value = 0.0
    amplitude = 1.0
    max_amplitude = 0.0
    frequency = 1.0 #higher frequency means smaller and more detailed 

    #  NOTE: check later 
    #   write out the match for how we are carving everything 
    for _ in range(octaves) :

        n = pnoise3(
        (x + 67) * scale * frequency,
        (y + 67) * scale * frequency,
        (z + 67) * scale * frequency,
        octaves=1, base=seed
        )

        # noise vals created at each octave
        value += amplitude * n 
        max_amplitude += amplitude
        amplitude *= persistance
        frequency *= lacunarity


    # Normalise values : 
    # each octave contributes amp * n with n E [-1, 1], 
    # max at every n = 1 means the sum(amplitudes) => dividing by this maps input to [-1,1]
    return value / max_amplitude


def gen_caves(values, ground_lvl, seed, density = 0.012, room_sz = 1.5, cave_val=0):
    gx, gy, gz = values.shape

    for ix in range(gx):
        for iy in range(gy):
            terrrain_top = int(ground_lvl[ix, iy])
            if terrrain_top<2:
                continue
            for iz in range(1, terrrain_top):
                
                effective_scale = 0.04 * room_sz # allow bigger rooms but keep all fine detail from higher octaves
                effective_octaves = max(2, int(4/room_sz)) #fewer octaves for bigger rooms
                noise_val = fractal_noise(ix, iy, iz, seed, scale=effective_scale,octaves=effective_octaves)
                threshold = density + (1.0 - iz / terrrain_top) * 0.02
                # depth biased threshol produces more caves near surface and fewer at depth (save power, deeper gen => bigger world gen )
                # to invert to deeper gen just remove the 1 - in the brackets (idk why you'd want to tho ngl)

                if noise_val > threshold:
                    values[ix, iy, iz] = cave_val
    return values





