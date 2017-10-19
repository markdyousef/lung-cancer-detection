import numpy as np

def make_mask(center, diameter, channels, width, height, spacing, origin):
    '''
        center: list of coordinates x,y,z
        diameter: float
        channels, width, height: pixel dim of image(z,y,x)
        spacing: nparray of coordinates x,y,z
        origin: nparray of coordinates x,y,z
    '''
    # 0 everywhere except nodule swapping x,y to macth img
    mask = np.zeros([height, width])

    # define voxel range for nodule
    v_center = (center-origin) / spacing
    v_diam = int(diameter / spacing[0]+5)
    v_xmin = np.max([0, int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1, int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0, int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1, int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin, v_xmax)
    v_yrange = range(v_ymin, v_ymax)

    # fill in 1 within sphere around module
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x, p_y, channels])) <= diameter:
                mask[int((p_y-origin[1]) / spacing[1]), int((p_x-origin[0]) / spacing[0])] = 1.0

    return mask
