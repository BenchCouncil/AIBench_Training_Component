euc_dist = function(p1, p2) {
    var x2 = Math.pow(p1.x - p2.x, 2),
        y2 = Math.pow(p1.y - p2.y, 2);
    return Math.sqrt(x2 + y2);
},


rand_pos = function(x_bounds, y_bounds) {

    var xrange = x_bounds.hi - x_bounds.lo,
        yrange = y_bounds.hi - y_bounds.lo;
    var px = Math.floor((Math.random()*(xrange))) + x_bounds.lo,
        py = Math.floor((Math.random()*(yrange))) + y_bounds.lo;
    return {x: px, y: py};
}

initialize_positions = function(num_obj, obj_radius, rand_pos_fn){
    var p0 = [];  // initial positions

    // set positions
    for (var i = 0; i < num_obj; i++) {

        // generate random initial positions by rejection sampling
        if (p0.length == 0) {  // assume that num_obj > 0
            p0.push(rand_pos_fn());
        } else {
            var proposed_pos = rand_pos_fn();
            // true if overlaps
            while ((function(){
                    for (var j = 0; j < p0.length; j++) {
                        if (euc_dist(proposed_pos, p0[j]) < 2.5*obj_radius) {
                            return true;
                        }
                    }
                    return false;
                })()){
                // keep trying until you get a match
                proposed_pos = rand_pos_fn();
            }
            p0.push(proposed_pos);
        }
    }
    return p0;
}


// sampled_sizes = [existing] + [will_sample]
// p0 --> [existing]
// num_obj --> [will_sample]
initialize_positions_variable_size_limited = function(num_obj, sampled_sizes, rand_pos_fn, p0, mul) {
    console.assert(num_obj + p0.length == sampled_sizes.length)
    console.log(num_obj + p0.length == sampled_sizes.length)

    // set positions
    for (let i = 0; i < num_obj; i++) {
        let num_iters = 0
        let should_break = false

        // generate random initial positions by rejection sampling
        if (p0.length == 0) {  // assume that num_obj > 0
            p0.push(rand_pos_fn());
        } else {
            let proposed_pos = rand_pos_fn();
            // true if overlaps
            while ((function(){
                    for (var j = 0; j < p0.length; j++) {
                        let other_size = sampled_sizes[j]  // this is the raw size, sizemul already incorporated. For an obstacle, let it be the diagonal from the center
                        let this_size = sampled_sizes[i]
                        let min_distance = other_size + this_size

                        if (euc_dist(proposed_pos, p0[j]) < mul*min_distance) {
                            num_iters ++
                            if (num_iters > 20) {
                                should_break = true
                                console.log('num_iters', num_iters, '> threshold. should_break set to true')
                                break
                            }
                            return true;
                        }
                    }
                    return false;
                })()){
                if (should_break) {
                    console.log('breaking')
                    break
                }
                // keep trying until you get a match
                proposed_pos = rand_pos_fn();
            }

            if (should_break) {
                console.log('broke. returning error signal')
                return [false, []]
            }
            p0.push(proposed_pos);
        }
    }
    return [true, p0];
}

initialize_positions_variable_size = function(num_obj, sampled_sizes, rand_pos_fn, p0, mul){
    let counter = 0
    let tolerance = 10000
    while (true) {
        console.log('proposing trajectory. counter:', counter)
        let p0_copy = p0.slice(0)
        let result = initialize_positions_variable_size_limited(num_obj, sampled_sizes, rand_pos_fn, p0_copy, mul)
        if (result[0]) {
            console.log('succeeded')
            return result[1]  // succeeded
        } 
        // otherwise keep going
        counter ++
        if (counter > tolerance) {
            console.log('Too hard to get a set of positions')
            assert(false)
            break
        }
    }
}

initialize_velocities = function(num_obj, max_v0) {
    var v0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial velocities b/w -max_v0 and max_v0 inclusive
        var vx = Math.floor((Math.random()*2*max_v0+1)-max_v0)
        var vy = Math.floor((Math.random()*2*max_v0+1)-max_v0)
        v0.push({x: vx, y: vy})
    }
    return v0;
}

initialize_hv = function(num_obj) {
    var a0 = [];
    for (var i = 0; i < num_obj; i++) {

        // generate random initial angles b/w -max_a0 and max_a0 inclusive
        var a = Math.floor(Math.random()*2)*Math.PI/2
        a0.push(a)
    }
    return a0;
}


initialize_masses = function(num_obj, possible_masses) {
    var masses = [];
    for (var i = 0; i < num_obj; i++) {

        // choose a random mass in the list of possible_masses
        var m = Math.floor(Math.random()*possible_masses.length)
        masses.push(possible_masses[m])
    }
    return masses;
}

initialize_sizes = function(num_obj, possible_sizes) {
    var sizes = [];
    for (var i = 0; i < num_obj; i++) {

        // choose a random mass in the list of possible_masses
        var s = Math.floor(Math.random()*possible_sizes.length)
        sizes.push(possible_sizes[s])
    }
    return sizes;
}

// assume trajectories ordered from bottom to top
is_stable_trajectory = function(trajectories) {
    // not stable if top block's y position is different it's original y position by a factor of a block length
    let top = trajectories[trajectories.length-1]
    let initial = {x: top[0].position.x, y: top[0].position.y}
    let final = {x: top[top.length-1].position.x, y: top[top.length-1].position.y}
    let dist = euc_dist(final, initial)
    // return dist


    let zdist = Math.abs(initial.y-final.y)
    return zdist

}

fraction_unstable = function(trajectories, stability_threshold) {
    // not stable if top block's y position is different it's original y position by a factor of a block length
    let num_fell = 0
    for (let i = 0; i < trajectories.length; i ++) {
        let obj = trajectories[i]
        let initial = {x: obj[0].position.x, y: obj[0].position.y}
        let final = {x: obj[obj.length-1].position.x, y: obj[obj.length-1].position.y}
        let dist = final.y - initial.y 
        console.log(i, dist)
        if (dist > stability_threshold) {
            num_fell ++
        }
    }
    console.log('Num fell', num_fell)
    return num_fell
}

// objects: array of bodies
// center of mass
// origin: demo.cx
// assume that bodies are ordered bottom to top
center_of_mass2 = function(bodies) {
    if (bodies.length == 1) {
        return bodies[0].position.x
    } else {
        let total_mass = 0
        let sum = 0
        let coms = []
        for (let i=0; i < bodies.length; i++) {
            if (i == 0) {
                coms.push(bodies[i].position.x)
            } else {
                // jth iteration: (prev)*((j-1)/j) + curr/j
                coms.push(coms[i-1]*(i)/(i+1) + bodies[i].mass*bodies[i].position.x/(i+1))  // zero indexed
            }
            total_mass += bodies[i].mass
            sum += bodies[i].mass*bodies[i].position.x
        }
        return coms
    }
}

center_of_mass = function(bodies) {
    if (bodies.length == 1) {
        return bodies[0].position.x
    } else {
        let total_mass = 0
        let sum = 0
        let coms = []
        for (let num_bodies=0; num_bodies < bodies.length; num_bodies++) {  // go from top most block down
            let curr_body = bodies[bodies.length-num_bodies-1]
            if (num_bodies == 0) {
                coms.push(curr_body.position.x)
            } else {
                // jth iteration: (prev)*((j-1)/j) + curr/j
                coms.push(coms[num_bodies-1]*(num_bodies)/(num_bodies+1) + curr_body.mass*curr_body.position.x/(num_bodies+1))  // zero indexed
            }
            total_mass += curr_body.mass
            sum += curr_body.mass*curr_body.position.x
        }
        return coms
    }
}

// random integer inclusive
random_int = function(lo, hi) {
    return lo+Math.floor((Math.random()*(hi-lo+1)))
}

convert_to_positions = function(array_of_arrays) {
    let array_of_positions = []
    for (let i=0; i < array_of_arrays.length; i ++) {
        array_of_positions.push({x:array_of_arrays[i][0], y:array_of_arrays[i][1]})
    }
    return array_of_positions
}



// Export
var _isBrowser = typeof window !== 'undefined' && window.location

if (!_isBrowser) {
    module.exports = function(){
        this.euc_dist = euc_dist
        this.rand_pos = rand_pos
        this.initialize_positions = initialize_positions
        this.initialize_velocities = initialize_velocities
        this.initialize_hv = initialize_hv
        this.initialize_positions_variable_size = initialize_positions_variable_size
        this.random_int = random_int
        this.convert_to_positions = convert_to_positions
    };
}


