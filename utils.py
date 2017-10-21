import numpy as np
from skimage import morphology, measure
from skimage.transform import resize
from sklearn.cluster import KMeans

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


def filter_lungs(prop):
    # background is removed by removing regions with a bbox to large
    # on either dimension
    # lungs are usually centered - remove regions to close to top and bottom
    bbox = prop.bbox
    if bbox[2]-bbox[0]-475 and bbox[3]-bbox[1]<475 and bbox[0]>40 and bbox[2]<472:
        return prop.label

def isolate_lungs(image_file):
    # load as float64 to work with scipy.kmeans
    imgs = np.load(image_file).astype(np.float64)
    for i in range(len(imgs)):
        img = imgs[i]
        # standardize px values
        mean = np.mean(img)
        std = np.std(img)
        img = (img-mean)/std
        # find avg px value near lungs (centered in scan)
        middle = img[100:400, 100:400]
        img_mean = np.mean(middle)
        img_max = np.max(img)
        img_min = np.min(img)
        # move underflow and overflow on px spectrum
        # improves threshold finding
        img[img==img_max]=mean
        img[img==img_min]=mean
        # kmeans for separating foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)
        # initial erosion to remove graininess for some regions
        # large dialation to make lung region engulf the vessels
        # and incursions into the lung cavity by radio-opaque tissue
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))

        # label regions and obtain region properties
        labels = measure.label(dilation)
        regions = measure.regionprops(labels)
        # retrieve labels for lungs
        good_labels = [filter_lungs(prop) for prop in regions]
        # mask lungs
        mask = np.ndarray([512, 512], dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N, 1, 0)
        # large dilation to fill out lung mask
        mask = morphology.dilation(mask, np.ones([10, 10]))
        imgs[i] = mask

    np.save(image_file.replace('images', 'lungmask'), imgs)

            
