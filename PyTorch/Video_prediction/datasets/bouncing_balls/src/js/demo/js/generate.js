/**
* Adapted from matter-js demo code
* The Matter.js demo page controller and example runner.
*
* NOTE: For the actual example code, refer to the source files in `/examples/`.
*
* @class Demo
*/

(function() {

    var _isBrowser = typeof window !== 'undefined' && window.location,
        _useInspector = _isBrowser && window.location.hash.indexOf('-inspect') !== -1,
        _isMobile = _isBrowser && /(ipad|iphone|ipod|android)/gi.test(navigator.userAgent),
        _isAutomatedTest = !_isBrowser || window._phantom;

    // var Matter = _isBrowser ? window.Matter : require('../../build/matter-dev.js');
    var Matter = _isBrowser ? window.Matter : require('matter-js');
    // var Matter = _isBrowser ? window.Matter : require('./ccd-matter')

    var Demo = {};
    Matter.Demo = Demo;

    if (!_isBrowser) {
        var jsonfile = require('jsonfile');
        var assert = require('assert');
        var utils = require('../../utils');
        var mkdirp = require('mkdirp');
        var fs = require('fs');
        require('./Examples')
        module.exports = Demo;
        window = {};
    }

    // Matter aliases
    var Body = Matter.Body,
        Example = Matter.Example,
        Engine = Matter.Engine,
        World = Matter.World,
        Common = Matter.Common,
        Composite = Matter.Composite,
        Bodies = Matter.Bodies,
        Events = Matter.Events,
        Mouse = Matter.Mouse,
        MouseConstraint = Matter.MouseConstraint,
        Runner = Matter.Runner,
        Render = Matter.Render;

    // MatterTools aliases
    if (window.MatterTools) {
        var Gui = MatterTools.Gui,
            Inspector = MatterTools.Inspector;
    }

    Demo.create = function(options) {
        var defaults = {
            isManual: false,
            sceneName: 'walls',
            sceneEvents: []
        };

        return Common.extend(defaults, options);
    };

    Demo.init = function(options) {
        var demo = Demo.create(options);
        Matter.Demo._demo = demo;

        demo.cmd_options = options

        // create an example engine (see /examples/engine.js)
        demo.engine = Example.engine(demo);


        if (_isBrowser) {
            // run the engine
            demo.runner = Engine.run(demo.engine);
            demo.runner.isFixed = true

            // get container element for the canvas
            demo.container = document.getElementById('canvas-container');  // this requires a browser

            // create a debug renderer
            demo.render = Render.create({
                element: demo.container,
                engine: demo.engine,
            });

            // run the renderer
            Render.run(demo.render);

            // add a mouse controlled constraint
            demo.mouseConstraint = MouseConstraint.create(demo.engine, {
                element: demo.render.canvas
            });

            World.add(demo.engine.world, demo.mouseConstraint);

            // pass mouse to renderer to enable showMousePosition
            demo.render.mouse = demo.mouseConstraint.mouse;

            // set up demo interface (see end of this file)
            Demo.initControls(demo);

            // get the scene function name from hash
            if (window.location.hash.length !== 0) {
                demo.sceneName = window.location.hash.replace('#', '').replace('-inspect', '');
            }

        } else {
            if (options.image) {
                // run the engine
                demo.runner = Runner.create()
                demo.runner.isFixed = true
                var pcanvas = PImage.make(800, 800);  // 693
                pcanvas.style = {}  
                console.log(pcanvas)
                demo.render = Render.create({
                    element: 17, // dummy
                    canvas: pcanvas,
                    engine: demo.engine,
                })
            }
        }

        // set up a scene with bodies
        Demo.reset(demo);

        if (_isBrowser)
            Demo.setScene(demo, demo.sceneName);

        // pass through runner as timing for debug rendering
        demo.engine.metrics.timing = demo.runner;

        return demo;
    };

    // call init when the page has loaded fully
    if (!_isAutomatedTest) {
        if (window.addEventListener) {
            window.addEventListener('load', Demo.init);
        } else if (window.attachEvent) {
            window.attachEvent('load', Demo.init);
        }
    }

    Demo.setScene = function(demo, sceneName) {
        Example[sceneName](demo);
    };

    // the functions for the demo interface and controls below
    Demo.initControls = function(demo) {
        var demoSelect = document.getElementById('demo-select'),
            demoReset = document.getElementById('demo-reset');

        // create a Matter.Gui
        if (!_isMobile && Gui) {
            demo.gui = Gui.create(demo.engine, demo.runner, demo.render);

            // need to add mouse constraint back in after gui clear or load is pressed
            Events.on(demo.gui, 'clear load', function() {
                demo.mouseConstraint = MouseConstraint.create(demo.engine, {
                    element: demo.render.canvas
                });

                World.add(demo.engine.world, demo.mouseConstraint);
            });
        }

        // create a Matter.Inspector
        if (!_isMobile && Inspector && _useInspector) {
            demo.inspector = Inspector.create(demo.engine, demo.runner, demo.render);

            Events.on(demo.inspector, 'import', function() {
                demo.mouseConstraint = MouseConstraint.create(demo.engine);
                World.add(demo.engine.world, demo.mouseConstraint);
            });

            Events.on(demo.inspector, 'play', function() {
                demo.mouseConstraint = MouseConstraint.create(demo.engine);
                World.add(demo.engine.world, demo.mouseConstraint);
            });

            Events.on(demo.inspector, 'selectStart', function() {
                demo.mouseConstraint.constraint.render.visible = false;
            });

            Events.on(demo.inspector, 'selectEnd', function() {
                demo.mouseConstraint.constraint.render.visible = true;
            });
        }

        // go fullscreen when using a mobile device
        if (_isMobile) {
            var body = document.body;

            body.className += ' is-mobile';
            demo.render.canvas.addEventListener('touchstart', Demo.fullscreen);

            var fullscreenChange = function() {
                var fullscreenEnabled = document.fullscreenEnabled || document.mozFullScreenEnabled || document.webkitFullscreenEnabled;

                // delay fullscreen styles until fullscreen has finished changing
                setTimeout(function() {
                    if (fullscreenEnabled) {
                        body.className += ' is-fullscreen';
                    } else {
                        body.className = body.className.replace('is-fullscreen', '');
                    }
                }, 2000);
            };

            document.addEventListener('webkitfullscreenchange', fullscreenChange);
            document.addEventListener('mozfullscreenchange', fullscreenChange);
            document.addEventListener('fullscreenchange', fullscreenChange);
        }

        // keyboard controls
        document.onkeypress = function(keys) {
            // shift + a = toggle manual
            if (keys.shiftKey && keys.keyCode === 65) {
                Demo.setManualControl(demo, !demo.isManual);
            }

            // shift + q = step
            if (keys.shiftKey && keys.keyCode === 81) {
                if (!demo.isManual) {
                    Demo.setManualControl(demo, true);
                }

                Runner.tick(demo.runner, demo.engine);
                console.log(demo.engine.world.bodies)
            }
        };

        // initialise demo selector
        demoSelect.value = demo.sceneName;
        Demo.setUpdateSourceLink(demo.sceneName);

        demoSelect.addEventListener('change', function(e) {
            Demo.reset(demo);
            Demo.setScene(demo,demo.sceneName = e.target.value);

            if (demo.gui) {
                Gui.update(demo.gui);
            }

            var scrollY = window.scrollY;
            window.location.hash = demo.sceneName;
            window.scrollY = scrollY;
            Demo.setUpdateSourceLink(demo.sceneName);
        });

        demoReset.addEventListener('click', function(e) {
            Demo.reset(demo);
            Demo.setScene(demo, demo.sceneName);

            if (demo.gui) {
                Gui.update(demo.gui);
            }

            Demo.setUpdateSourceLink(demo.sceneName);
        });
    };

    Demo.setUpdateSourceLink = function(sceneName) {
        var demoViewSource = document.getElementById('demo-view-source'),
            sourceUrl = 'https://github.com/liabru/matter-js/blob/master/examples';
        demoViewSource.setAttribute('href', sourceUrl + '/' + sceneName + '.js');
    };

    Demo.setManualControl = function(demo, isManual) {
        var engine = demo.engine,
            world = engine.world,
            runner = demo.runner;

        demo.isManual = isManual;

        if (demo.isManual) {
            Runner.stop(runner);

            // continue rendering but not updating
            (function render(time){
                runner.frameRequestId = window.requestAnimationFrame(render);
                Events.trigger(engine, 'beforeUpdate');
                Events.trigger(engine, 'tick');
                engine.render.controller.world(engine);  // should be called every time a scene changes
                Events.trigger(engine, 'afterUpdate');
            })();
        } else {
            Runner.stop(runner);
            Runner.start(runner, engine);
        }
    };

    Demo.fullscreen = function(demo) {
        var _fullscreenElement = demo.render.canvas;

        if (!document.fullscreenElement && !document.mozFullScreenElement && !document.webkitFullscreenElement) {
            if (_fullscreenElement.requestFullscreen) {
                _fullscreenElement.requestFullscreen();
            } else if (_fullscreenElement.mozRequestFullScreen) {
                _fullscreenElement.mozRequestFullScreen();
            } else if (_fullscreenElement.webkitRequestFullscreen) {
                _fullscreenElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
            }
        }
    };

    Demo.reset = function(demo) {
        var world = demo.engine.world,
            i;

        World.clear(world);
        Engine.clear(demo.engine);

        // clear scene graph (if defined in controller)
        if (demo.render) {
            var renderController = demo.render.controller;
            if (renderController && renderController.clear)
                renderController.clear(demo.render);
        }

        // clear all scene events
        if (demo.engine.events) {
            for (i = 0; i < demo.sceneEvents.length; i++)
                Events.off(demo.engine, demo.sceneEvents[i]);
        }

        if (demo.mouseConstraint && demo.mouseConstraint.events) {
            for (i = 0; i < demo.sceneEvents.length; i++)
                Events.off(demo.mouseConstraint, demo.sceneEvents[i]);
        }

        if (world.events) {
            for (i = 0; i < demo.sceneEvents.length; i++)
                Events.off(world, demo.sceneEvents[i]);
        }

        if (demo.runner && demo.runner.events) {
            for (i = 0; i < demo.sceneEvents.length; i++)
                Events.off(demo.runner, demo.sceneEvents[i]);
        }

        if (demo.render && demo.render.events) {
            for (i = 0; i < demo.sceneEvents.length; i++)
                Events.off(demo.render, demo.sceneEvents[i]);
        }

        demo.sceneEvents = [];

        // reset id pool
        Body._nextCollidingGroupId = 1;
        Body._nextNonCollidingGroupId = -1;
        Body._nextCategory = 0x0001;
        Common._nextId = 0;

        // reset random seed
        Common._seed = 0;

        // reset mouse offset and scale (only required for Demo.views)
        if (demo.mouseConstraint) {
            Mouse.setScale(demo.mouseConstraint.mouse, { x: 1, y: 1 });
            Mouse.setOffset(demo.mouseConstraint.mouse, { x: 0, y: 0 });
        }

        demo.engine.enableSleeping = false;
        demo.engine.world.gravity.y = 1;    // default
        demo.engine.world.gravity.x = 0;
        demo.engine.timing.timeScale = 1;

        demo.offset = 5;  // world offset
        demo.config = {}
        demo.config.cx = 400;
        demo.config.cy = 400;
        demo.config.masses = [1, 5, 25]
        demo.config.mass_colors = {'1':'#C7F464', '5':'#FF6B6B', '25':'#4ECDC4'}
        demo.config.sizes = [2/3, 1, 3/2]  // multiples
        demo.config.drastic_sizes = [1/2, 1, 2]  // multiples 
        demo.config.object_base_size = {'ball': 60, 'obstacle': 80, 'block': 20 }  // radius of ball, side of square obstacle, long side of block
        demo.config.objtypes = ['ball', 'obstacle', 'block']  // squares are obstacles
        demo.config.g = 0 // The index of the one hot. 0 is no, 1 is yes
        demo.config.f = 0 // 
        demo.config.p = 0 // 
        demo.config.max_velocity = 60

        demo.cx = demo.config.cx;
        demo.cy = demo.config.cy;
        demo.width = 2*demo.cx
        demo.height = 2*demo.cy

        // this is correct
        demo.engine.world.bounds = { min: { x: 0, y: 0 },
                                    max: { x: demo.width, y: demo.height }} 

        if (demo.cmd_options.image) {
            demo.render.hasBounds = true
            demo.render.options.height = demo.height
            demo.render.options.width = demo.width
            demo.render.canvas.height = demo.height
            demo.render.canvas.width = demo.width
        }

        if (demo.mouseConstraint) {
            World.add(world, demo.mouseConstraint);
        }

        if (demo.render) {
            var renderOptions = demo.render.options;
            renderOptions.wireframes = false;
            renderOptions.hasBounds = false;
            renderOptions.showDebug = false;
            renderOptions.showBroadphase = false;
            renderOptions.showBounds = false;
            renderOptions.showVelocity = false;
            renderOptions.showCollisions = false;
            renderOptions.showAxes = false;
            renderOptions.showPositions = false;
            renderOptions.showAngleIndicator = false;
            renderOptions.showIds = false;
            renderOptions.showShadows = false;
            renderOptions.showVertexNumbers = false;
            renderOptions.showConvexHulls = false;
            renderOptions.showInternalEdges = false;
            renderOptions.showSeparations = false;
            renderOptions.background = '#fff';

            if (_isMobile) {
                renderOptions.showDebug = true;
            }
        }
    };

    Demo.simulate = function(demo, num_samples, sim_options, startstep) {
        var trajectories = []

        console.log(sim_options)
        if (sim_options.env == 'tower') {
            var num_unstable = 0
            var num_stable = 0
            var total_fell = 0
            var stability_threshold = 5
        }

        if (!(typeof startstep !== 'undefined' &&  startstep)) {
            var startstep = 0
        }

        let s = startstep;
        console.log("Start step:", s)
        while (s < num_samples) {
            Demo.reset(demo);
            var scenario = Example[sim_options.env](demo, sim_options)
            var trajectory = []
            console.log(s)


            // initialize trajectory conatiner
            for (id = 0; id < scenario.params.num_obj; id++) { //id = 0 corresponds to world
                trajectory[id] = [];
            }

            // Now iterate through all ids to find which ones have the "Entity" label, store those ids
            var entities = Composite.allBodies(scenario.engine.world)
                            .filter(function(elem) {
                                        return elem.label === 'Entity';
                                    })

            var entity_ids = entities.map(function(elem) {
                                return elem.id});

            assert(entity_ids.length == scenario.params.num_obj)

            var should_break = false
            // run the engine
            for (let i = 0; i < sim_options.steps; i++) {


                for (let id = 0; id < scenario.params.num_obj; id++) { //id = 0 corresponds to world
                    trajectory[id][i] = {};
                    let body = Composite.get(scenario.engine.world, entity_ids[id], 'body')
                    for (let k of ['position', 'velocity', 'mass', 'angle', 'angularVelocity', 'objtype', 'sizemul']){
                        trajectory[id][i][k] = utils.copy(body[k])

                        // check if undefined.
                        if (!(typeof trajectory[id][i][k] !== 'undefined')) {
                            should_break = true;
                            console.log('trajectory[id][i][k] is undefined', trajectory[id][i][k])
                            console.log('id',id,'i',i,'k',k)
                            break;
                        }
                    }
                    if (should_break) {break;}

                    // check for invalid conditions
                    if (Math.abs(trajectory[id][i]['velocity'].x) > demo.config.max_velocity || Math.abs(trajectory[id][i]['velocity'].y) > demo.config.max_velocity) {
                        should_break = true;
                        console.log('Set should_break to true. max velocity', demo.config.max_velocity)
                        console.log('this velocity', trajectory[id][i]['velocity'])
                        break;
                    }
                    if (trajectory[id][i]['position'].x > demo.width || trajectory[id][i]['position'].x < 0 ||
                        trajectory[id][i]['position'].y > demo.height || trajectory[id][i]['position'].y < 0) {
                        should_break = true;
                        console.log('Set should_break to true. demo.engine.world.bounds', demo.engine.world.bounds)
                        console.log('this position', trajectory[id][i]['position'])
                        break;
                    }
                }
                if (should_break) {break;}

                Engine.update(scenario.engine);

                if (sim_options.image) {
                    demo.render.context.fillStyle = 'white'
                    demo.render.context.fillRect(0,0,demo.width,demo.height)
                    Render.world(demo.render)
                }

                // if (sim_options.env == 'tower') {
                //     if (i == 59) {
                //         console.log('euc dist', i, is_stable_trajectory(trajectory))
                //         console.log('stable?', i, is_stable_trajectory(trajectory) < stability_threshold)
                //     } else if (i == 119) {
                //         console.log('euc dist', i, is_stable_trajectory(trajectory))
                //         console.log('stable?', i, is_stable_trajectory(trajectory) < stability_threshold)
                //     } 
                // }
            }

            if (should_break) {
                console.log('Break. Trying again.')
            } else {  // valid trajectory
                if (sim_options.env == 'tower') {
                    // if (is_stable_trajectory(trajectory) > stability_threshold) {

                    let num_fell = fraction_unstable(trajectory, 1)
                    if (num_fell > 0) {
                        num_unstable ++
                    } else {
                        num_stable ++
                    }
                    // if (scenario.stable) {
                    //     com_num_stable ++
                    // } else {
                    //     com_num_unstable ++
                    // }
                    total_fell = total_fell + num_fell

                    if (num_stable > num_samples/2) {
                        console.log('num_stable > num_samples/2. Want more unstable')
                        num_stable -- 
                        continue
                    } else if (num_unstable > num_samples/2) {
                        console.log('num_unstable > num_samples/2. Want more stable')
                        num_unstable --
                        continue
                    }
                }
                trajectories[s] = trajectory;  // basically I can't reach this part
                s++;
            }
        }

        // let frac_fell = total_fell / (num_samples * scenario.params.num_obj)

        if (sim_options.env == 'tower') {
            // console.log(num_unstable, 'unstable threshold', com_num_unstable, 'unstable com out of', num_samples, 'samples')
            return [trajectories, num_unstable, total_fell];  // NOTE TOWER
        } else {
            return trajectories
        }

    };

    Demo.create_json_fname = function(samples, id, sim_options) {  // later add in the indices are something
        // experiment string
        let experiment_string = sim_options.env +
                                '_n' + sim_options.numObj +
                                '_t' + sim_options.steps +
                                '_ex' + sim_options.samples

        if (sim_options.variableMass) {
            experiment_string += '_m' 
        }
        if (sim_options.variableSize) {
            experiment_string += '_z' 
        }
        if (sim_options.variableObstacles) {
            experiment_string += '_o' 
        }
        if (sim_options.drasticSize) {
            experiment_string += '_dras3'
        }
        if (typeof sim_options.wall !== 'undefined' && sim_options.wall) {
            experiment_string += '_w' + sim_options.wall
        }
        if (sim_options.gravity) {
            experiment_string += '_gf'
        }
        if (sim_options.pairwise) {
            experiment_string += '_pf'
        }
        if (sim_options.friction) {
            experiment_string += '_fr'
        }
        var savefolder = '../../data/' + experiment_string + '/jsons/'

        if (!fs.existsSync(savefolder)){
            mkdirp.sync(savefolder);
        }

        experiment_string += '_chksize' + samples + '_' + id

        let sim_file = savefolder + experiment_string + '.json';
        return sim_file;
    };

    Demo.generate_data = function(demo, sim_options) {
        // const max_iters_per_json// = 100; 
        let max_iters_per_json
        if (sim_options.env == 'walls') {
            max_iters_per_json = 20
        } else {
            max_iters_per_json = 100
        }

        if (!(typeof sim_options.startstep !== 'undefined' &&  sim_options.startstep)) {
            sim_options.startstep = 0
        }

        const chunks = chunk(sim_options.samples, max_iters_per_json, sim_options.startstep)
        let num_examples_left = chunks.reduce(function(a, b) { return a + b; }, 0);

        if (sim_options.startstep < sim_options.samples) {
            // tower
            if (sim_options.env == 'tower') {
                var num_unstable = 0
                // var num_comunstable = 0
                var num_total_fell = 0
                var num_total_samples = 0
            }

            for (let j=0; j < chunks.length; j++){
                let chunk_number = j + (sim_options.samples-num_examples_left)/max_iters_per_json

                let sim_file = Demo.create_json_fname(chunks[j], chunk_number, sim_options)

                if (sim_options.env == 'tower') {
                    let output = Demo.simulate(demo, chunks[j], sim_options, sim_options.startstep);
                    let trajectories = output[0]
                    let num_unstable_chunk = output[1]
                    // let com_num_unstable_chunk = output[2]
                    let num_total_fell_chunk = output[2]
                    num_unstable += num_unstable_chunk
                    // num_comunstable += com_num_unstable_chunk
                    num_total_fell += num_total_fell_chunk
                    num_total_samples += chunks[j]

                    jsonfile.writeFileSync(sim_file,
                                        {trajectories:trajectories, config:sim_options}
                                        );
                    console.log('Wrote to ' + sim_file)
                } else {
                    let trajectories = Demo.simulate(demo, chunks[j], sim_options);

                    jsonfile.writeFileSync(sim_file,
                                        {trajectories:trajectories, config:sim_options}
                                        );
                    console.log('Wrote to ' + sim_file)
                }

                // let unstable_counts = Demo.generate_chunk(chunks, chunk_number, sim_options, num_unstable, num_comunstable)
                // num_unstable = unstable_counts[0]
                // num_comunstable = unstable_counts[1]
            }


            // tower
            if (sim_options.env == 'tower') {
                console.log(num_unstable, 'unstable threshold', num_total_fell/(num_total_samples*sim_options.numObj), 'fraction fell out of', sim_options.samples, 'samples')
            }

        }

    };

    Demo.process_cmd_options = function() {
        const optionator = require('optionator')({
            options: [{
                    option: 'help',
                    alias: 'h',
                    type: 'Boolean',
                    description: 'displays help',
                }, {
                    option: 'env',
                    alias: 'e',
                    type: 'String',
                    description: 'base environment',
                    required: true
                }, {
                    option: 'numObj',
                    alias: 'n',
                    type: 'Int',
                    description: 'number of objects',
                    required: true
                }, {
                    option: 'steps',
                    alias: 't',
                    type: 'Int',
                    description: 'number of timesteps',
                    required: true
                }, {
                    option: 'samples',
                    alias: 's',
                    type: 'Int',
                    description: 'number of samples',
                    required: true
                }, {
                    option: 'variableMass',
                    alias: 'm',
                    type: 'Boolean',
                    description: 'include variable mass',
                    required: false
                }, {
                    option: 'variableSize',
                    alias: 'z',
                    type: 'Boolean',
                    description: 'include variable size',
                    required: false
                }, {
                    option: 'variableObstacles',
                    alias: 'o',
                    type: 'Boolean',
                    description: 'true if number of obstacles is variable',
                    required: false
                }, {
                    option: 'drasticSize',
                    alias: 'd',
                    type: 'Boolean',
                    description: 'true if size difference is drastic',
                    required: false
                }, {
                    option: 'wall',
                    alias: 'w',
                    type: 'String',
                    description: 'wall type O | L | U | I',
                    required: false
                }, {
                    option: 'image',
                    alias: 'i',
                    type: 'Boolean',
                    description: 'include image frames',
                    default: false
                }, {
                    option: 'gravity',
                    alias: 'g',
                    type: 'Boolean',
                    description: 'include gravity',
                    default: false
                }, {
                    option: 'friction',
                    alias: 'f',
                    type: 'Boolean',
                    description: 'include friction',
                    default: false
                }, {
                    option: 'pairwise',
                    alias: 'p',
                    type: 'Boolean',
                    description: 'include pairwise forces',
                    default: false
                }, {
                    option: 'startstep',
                    alias: 'y',
                    type: 'Int',
                    description: 'starting step to generation',
                    default: false
                }]
        });

        // process invalid optiosn
        try {
            optionator.parseArgv(process.argv);
        } catch(e) {
            console.log(optionator.generateHelp());
            console.log(e.message)
            process.exit(1)
        }

        const cmd_options = optionator.parseArgv(process.argv);
        console.log(cmd_options)
        if (cmd_options.help) console.log(optionator.generateHelp());
        return cmd_options;
    };

    // main
    if (!_isBrowser) {
        const cmd_options = Demo.process_cmd_options();
        console.log('processed command options')
        var demo = Demo.init(cmd_options)  // don't set the scene name yet
        console.log('initialized. generating data')
        Demo.generate_data(demo, cmd_options);
    }
})();
