import {assertValidResolution, toValidInputResolution} from './util';

describe('util', () => {
  describe('toValidInputResolution', () => {
    it('returns an odd value', () => {
      expect(toValidInputResolution(1920, 8) % 2).toEqual(1);
      expect(toValidInputResolution(1280, 16) % 2).toEqual(1);
      expect(toValidInputResolution(719, 16) % 2).toEqual(1);
      expect(toValidInputResolution(545, 16) % 2).toEqual(1);
      expect(toValidInputResolution(225, 8) % 2).toEqual(1);
      expect(toValidInputResolution(240, 8) % 2).toEqual(1);
    });

    it('returns the original value when it is already a valid resolution',
       () => {
         const outputStride = 16;

         const validResolution = toValidInputResolution(1000, outputStride);

         const resolution =
             toValidInputResolution(validResolution, outputStride);

         expect(resolution).toEqual(validResolution);
       });

    it('returns a value that when 1 is subtracted by it is ' +
           'divisible by the output stride',
       () => {
         const outputStride = 8;
         const inputResolution = 562;

         const resolution =
             toValidInputResolution(inputResolution, outputStride);

         expect((resolution - 1) % outputStride).toEqual(0);
       });
  });

  describe('assertValidResolution', () => {
    it('raises an error when one subtracted by the input resolution is ' +
           'not divisible by the output stride',
       () => {
         expect(() => {
           assertValidResolution(
               [
                 toValidInputResolution(16 * 5, 16) + 1,
                 toValidInputResolution(16 * 5, 16)
               ],
               16);
         }).toThrow();
       });
    it('does not raise an error when one subtracted by the input resolution is ' +
           'divisible by the output stride',
       () => {
         expect(() => {
           assertValidResolution(
               [
                 toValidInputResolution(16 * 10, 16),
                 toValidInputResolution(16 * 5 + 20, 16)
               ],
               16);
         }).not.toThrow();
         expect(() => {
           assertValidResolution(
               [
                 toValidInputResolution(20 * 5, 32),
                 toValidInputResolution(5 * 30 + 20, 32)
               ],
               32);
         }).not.toThrow();
       });
  });
});
