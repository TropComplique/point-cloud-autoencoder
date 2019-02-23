from plyfile import PlyData



def load_ply(filename):
    ply_data = PlyData.read(filename)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    return points


def random_rotation_matrix(deflection=1.0):
    """Creates a random rotation matrix.
    Arguments:
        deflection: the magnitude of the rotation. 
        For 0, no rotation; for 1, completely random
        rotation. Small deflection => small perturbation.
    Returns:
        a numpy float array with shape [3, 3].
    """

    theta, phi, z = np.random.uniform(size=[3])

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = [
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    ]

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    r_rotation[0, 2] = 0
    r_rotation[2, 0] = 0
    r_rotation[1, 2] = 0
    r_rotation[2, 1] = 0
    r_rotation[2, 2] = 1

    return M





        
def random_shift(cloud, mu=0, std=1):
    gnoise = np.random.normal(mu, std, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def apply_augmentations(clouds):
    
    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        
        batch = batch.dot(r_rotation)
    return batch

snc_synth_id_to_category = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}