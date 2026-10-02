// Laser depos -> PDHD ADC frames, with anode 0's field response an argument.
//
//     wire-cell -l stdout -L info -V elecGain=14 \
//         -A depos=runs/depos/laser.npz -A frames=runs/frames/data.npz \
//         -A fr_bad=<FR for anode 0> -A fr_good=dune-garfield-1d565.json.bz2 \
//         -A anodes='[0,2]' -A noise=false -A fluctuate=false \
//         -c sim-laser.jsonnet
//
// DepoFileSource (one depo set per laser shot) -> DepoSetDrifter ->
// DepoSetFanout over the chosen anodes -> per anode: DepoTransform (PIR from
// that anode's FR) -> Reframer -> [AddNoise] -> Digitizer -> FrameFanin ->
// FrameFileSink.
//
// Built on pgrapher/experiment/pdhd/{simparams,sim}.jsonnet.  Their
// files.fields is one FR per anode, and sim.jsonnet uses tools.pirs[n] for
// tools.anodes[n], so we override fields from the arguments and select the
// anodes by filtering anodes and pirs together.
//
// Arguments:
//   depos      input depo file (.npz), as made by scripts/laser_depos.py
//   frames     output frame file (.npz); frames are tagged orig<ident>
//   fr_bad     FR for anode 0: the truth, a prior, or a trial written by a fit
//   fr_good    FR for anodes 1-3
//   anodes     JSON list of anode idents to simulate
//   noise      "true" adds the PDHD empirical noise
//   fluctuate  "true" enables Poisson/binomial charge fluctuations
//   seeds      JSON list of 5 pRNG seeds
//   digitize   "false" skips the digitizer and writes the analog (voltage)
//              frames as float, for linear use (learner 2); no noise then

local wc = import "wirecell.jsonnet";
local g = import "pgraph.jsonnet";
local f = import "pgrapher/common/funcs.jsonnet";
local io = import "fileio.jsonnet";
local tools_maker = import "pgrapher/common/tools.jsonnet";
local simparams = import "pgrapher/experiment/pdhd/simparams.jsonnet";
local sim_maker = import "pgrapher/experiment/pdhd/sim.jsonnet";

local tobool(s) = if std.isBoolean(s) then s else std.asciiLower(s) == "true";
local tolist(s) = if std.isString(s) then std.parseJson(s) else s;

function(depos="depos.npz", frames="frames.npz",
         fr_bad="np04hd-garfield-6paths-mcmc-bestfit.json.bz2",
         fr_good="dune-garfield-1d565.json.bz2",
         anodes="[0,2]", noise="false", fluctuate="false", seeds="[0,1,2,3,4]",
         digitize="true")

    local params = simparams {
        files: super.files {
            fields: [fr_bad, fr_good, fr_good, fr_good],
        },
        sim: super.sim {
            fluctuate: tobool(fluctuate),
        },
    };

    local all_tools = tools_maker(params);
    local idents = tolist(anodes);
    local keep = [n for n in std.range(0, std.length(all_tools.anodes) - 1)
                  if std.member(idents, all_tools.anodes[n].data.ident)];
    local tools = all_tools {
        anodes: [all_tools.anodes[n] for n in keep],
        pirs: [all_tools.pirs[n] for n in keep],
        random: super.random { data: super.data { seeds: tolist(seeds) } },
    };
    local sim = sim_maker(params, tools);

    local source = io.depo_file_source(depos);
    local setdrifter = g.pnode({
        type: "DepoSetDrifter",
        data: { drifter: "Drifter" },   // sim.drifter is an unnamed Drifter
    }, nin=1, nout=1, uses=[sim.drifter]);

    // Analog: the same DepoTransform and Reframer as pdhd/sim.jsonnet, no
    // digitizer.
    local analog_pipes = [g.pipeline([
        sim.make_depotransform("depotransform-" + tools.anodes[n].name,
                               tools.anodes[n], tools.pirs[n]),
        g.pnode({
            type: "Reframer",
            name: "reframer-" + tools.anodes[n].name,
            data: {
                anode: wc.tn(tools.anodes[n]),
                tags: [],
                fill: 0.0,
                tbin: params.sim.reframer.tbin,
                toffset: 0,
                nticks: params.sim.reframer.nticks,
            },
        }, nin=1, nout=1)], name="simanalogpipe-" + tools.anodes[n].name)
        for n in std.range(0, std.length(tools.anodes) - 1)];

    local pipes = if !tobool(digitize) then analog_pipes
                  else if tobool(noise) then sim.splusn_pipelines
                  else sim.signal_pipelines;
    local outtags = ["orig%d" % a.data.ident for a in tools.anodes];
    local fanpipe = f.fanpipe("DepoSetFanout", pipes, "FrameFanin", "simlaser", outtags);

    local sink = io.frame_file_sink(frames, tags=outtags, digitize=tobool(digitize));

    local graph = g.pipeline([source, setdrifter, fanpipe, sink]);
    local app = {
        type: "Pgrapher",
        data: { edges: g.edges(graph) },
    };
    local cmdline = {
        type: "wire-cell",
        data: {
            plugins: ["WireCellGen", "WireCellPgraph", "WireCellSio", "WireCellAux",
                      "WireCellSigProc"],
            apps: ["Pgrapher"],
        },
    };
    [cmdline] + g.uses(graph) + [app]
