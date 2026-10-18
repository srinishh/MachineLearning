import * as THREE from 'three';
import { sphereCloud, boxCloud, setInstance, setInstanceColor, commitInstances } from '../helpers.js';
import { IS_TEST } from '../data/covid.js';

// Data points drawn as spheres (training set) or cubes (test set). Both meshes
// hold every index; the unused one is scaled to zero.
export function splitPoints(count, radius = 0.13, isTest = IS_TEST) {
  const train = sphereCloud(count, radius);
  const test = boxCloud(count, radius * 1.6);
  const group = new THREE.Group();
  group.add(train, test);
  return {
    group,
    set(i, x, y, z, scale = 1, color) {
      const own = isTest[i] ? test : train;
      setInstance(own, i, x, y, z, scale);
      setInstance(isTest[i] ? train : test, i, 0, 0, 0, 0);
      if (color !== undefined) setInstanceColor(own, i, color);
    },
    commit() {
      commitInstances(train);
      commitInstances(test);
    },
    // Index of the clicked point, or -1.
    pick(ray) {
      const hit = ray.intersectObjects([train, test])[0];
      return hit?.instanceId ?? -1;
    },
  };
}

export const SPLIT_LEGEND = '<div class="legend split-legend"><span class="shape-sphere">● train (sphere)</span><span class="shape-cube">■ test (cube)</span></div>';
