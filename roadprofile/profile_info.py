import numpy as np

def profile_info(meter, z):
    mm = meter * 1000
    dist_points = np.diff(mm)
    monotone = 'NO'
    if all(dist_points > 0):
        monotone = 'YES'
    coinciding_points = 'YES'
    if all(dist_points != 0):
        coinciding_points = 'NO'
    mean_dist = np.mean(dist_points)
    std_dist = np.std(dist_points)
    largest_dist = max(dist_points)
    smallest_dist = min(dist_points)

    text = [
        'Number of measurement points:\t {:2.2f} mio.'.format(len(meter)/1e6),
        'Length of measured sections:\t {:2.2f} km'.format((meter[-1] - meter[0])/1000),
        'Measurement increasing: {}'.format(monotone),
        'Any measurements coinciding: {}'.format(coinciding_points),
        'Mean distance between points:\t\t {:1.4f} mm (σ = {:1.4f} mm)'.format(mean_dist, std_dist),
        'Largest distance between points:\t {:1.4f} mm'.format(largest_dist),
        'Smallest distance between points:\t {:1.4f} mm'.format(smallest_dist),
        ]
    print('\n'.join(text))
