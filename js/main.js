const ort = require('onnxruntime-web');

let encoder_sess;
let decoder_sess;

encoderModel = "encoder.model.ort";
decoderModel = "decoder.model.ort";

let output_alphabet = {0: "SOS", 1: "EOS", 2: "m", 3: "ɝ", 4: "ə", 5: "k", 6: "ɪ", 7: "b", 8: "a", 9: "ʊ", 10: "t", 11: "ɒ", 12: "z", 13: "ɔ", 14: "d", 15: "iː", 16: "n", 17: "w", 18: "uː", 19: "s", 20: "ɛ", 21: "l", 22: "v", 23: "f", 24: "ɔː", 25: "ɹ", 26: "e", 27: "ɡ", 28: "æ", 29: "m̩", 30: "ʌ", 31: "ŋ", 32: "p", 33: "j", 34: "θ", 35: "ɑː", 36: "u", 37: "n̩", 38: "t͡ʃ", 39: "ɪ̯", 40: "ɑ", 41: "d͡ʒ", 42: "i", 43: "h", 44: "ɛː", 45: "o", 46: "l̩", 47: "ɚ", 48: "ʃ", 49: "ʒ", 50: "ɜː", 51: "əː", 52: "aː", 53: "r", 54: "eː", 55: "ʔ", 56: "æː", 57: "ɫ", 58: "ɜ", 59: "ð", 60: "oː", 61: "ʊ̯", 62: "ɪː", 63: "ʍ", 64: "ɝː"}


document.addEventListener("DOMContentLoaded", function () {

    let process= document.getElementById('process');

    process.addEventListener("click", () => {
        processInput();
    });

    console.log("loading encoder model");
    console.log(ort.env)
    console.log(ort.env.wasm)
    try {
        encoder_sess = new Encoder(encoderModel, (e) => {
            if (e === undefined) {
                console.log(`${encoderModel} loaded, ${ort.env.wasm.numThreads} threads`);
                //ready();
            } else {
                console.log(`Error: ${e}`);
            }
        });

    } catch (e) {
        console.log(`Error: ${e}`);
    }

    console.log("loading decoder model");
    console.log(ort.env)
    console.log(ort.env.wasm)
    try {
        decoder_sess = new Decoder(decoderModel, (e) => {
            if (e === undefined) {
                console.log(`${decoderModel} loaded, ${ort.env.wasm.numThreads} threads`);
                //ready();
            } else {
                console.log(`Error: ${e}`);
            }
        });

    } catch (e) {
        console.log(`Error: ${e}`);
    }


})

class Decoder {
    constructor(url, cb) {
        ort.env.logLevel = "error";
        this.sess = null;

        const opt = {
            executionProviders: ["wasm"],
            logSeverityLevel: 3,
            logVerbosityLevel: 3,
        };
        ort.InferenceSession.create(url, opt).then((s) => {
            this.sess = s;
            cb();
        }, (e) => { cb(e); })
    }

    async run(decoder_input, decoder_hidden, beams = 1) {
        console.log('Decoder::run()');
        // clone semi constants into feed. The clone is needed if we run with ort.env.wasm.proxy=true
        const feed = {
            "l_encoder_outputs_": decoder_input,
            "l_encoder_hidden_": decoder_hidden
        }

        console.log(feed);
        return this.sess.run(feed);
    }
}


class Encoder {
    constructor(url, cb) {
        ort.env.logLevel = "error";
        this.sess = null;

        const opt = {
            executionProviders: ["wasm"],
            logSeverityLevel: 3,
            logVerbosityLevel: 3,
        };
        ort.InferenceSession.create(url, opt).then((s) => {
            this.sess = s;
            cb();
        }, (e) => { cb(e); })
    }

    async run(encoder_input, beams = 1) {
        console.log('Encoder::run()');
        // clone semi constants into feed. The clone is needed if we run with ort.env.wasm.proxy=true
        const feed = {
            "l_input_": encoder_input
        }
        console.log(feed);

        return this.sess.run(feed);
    }
}

function argMax(array) {
  return [].map.call(array, (x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

async function processInput() {
    console.log('processInput()');
    let outputSpan = document.getElementById('output');
    try {
          var input_array = new Int32Array([ 8, 15, 11,  6,  9,  6, 10, 11,  6,  7,  1]);
          var encoder_input = new ort.Tensor(input_array, [1, 11]); // reshape the input
          const ret = await encoder_sess.run(encoder_input);
          console.log(ret);
          console.log(ret.gru_1); 
          console.log(ret.gru_1_1);
          const ret2 = await decoder_sess.run(ret.gru_1, ret.gru_1_1);
          console.log(ret2);
          console.log(ret2._log_softmax[0]);

          // get the argmax of the softmax
          var argmaxes = Array();
          for(var  i = 0; i < ret2._log_softmax.cpuData.length; i += 65) {
            //console.log('i=' + i);
            var vrow = Array();
            for(var j = 0; j < 20; j++) {
              //console.log('j=' + j);
              let idx = i+j;
              //console.log('idx=' + idx);
              vrow.push(ret2._log_softmax.cpuData[idx]);
            }
            console.log(vrow);
            max_idx = argMax(vrow);
            console.log('argMax=' + max_idx);
            argmaxes.push(max_idx);
          }
          for(var i = 0; i < argmaxes.length; i++) {
              var sym = argmaxes[i];
              console.log('MAX: ' + sym + " " + output_alphabet[sym]);
              outputSpan.innerHTML += output_alphabet[sym] + " ";
          }

        } catch (e) {
            console.log(`Error: ${e}`);
        }

}

