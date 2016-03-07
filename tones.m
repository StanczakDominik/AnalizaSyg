function [frequency,name]=tones(fminHz,fmaxHz)
%Author:    Teodor Buchner
%           Faculty of Physics,Warsaw University of Technology, Warsaw, Poland
%E-mail:    buchner@if.pw.edu.pl
%           
%Create:    2010.04.27 01:28
%Brief introduction:Return [frequency,name]: tones of equally tempered scale and their names for a frequency range.
%Variable:  fminHz - minimum frequency
%           fmaxHz - maximum frequency
%-------------------------------------------------
%Make sure x and y are with same form.
%C0 	 16.35
%C1      32.70
%C2 	 65.41
%C3 	 130.81
%C4 	 261.63
%C5 	 523.25
%C6 	 1046.50
%C7 	 2093.00
%C8 	 4186.01
%A0 	 27.50
%A1 	 55.00
%A2 	 110.00
%A3 	 220.00
%A4 	 440.00
%A7 	 3520.00
% For comparison: The fundamental pitch of note.wav.
% keyhole C => 9 semitones below 440 Hz "A"
%fActual_Hz=440*power(2, -9/12)
%a=sin(2*pi*440*x);
%b=sin(2*pi*493.88*x);
%cs=sin(2*pi*554.37*x);
%d=sin(2*pi*587.33*x);
%e=sin(2*pi*659.26*x);
%fs=sin(2*pi*739.99*x);
min_note_distance = -57;
max_note_distance = 39;
printf("Version 1");
siz=max_note_distance - min_note_distance
frequency = zeros(1,1+max_note_distance - min_note_distance);
name = {1,1+max_note_distance - min_note_distance};
%base_frequency = [ 16.35 32.70 65.41 130.81 261.63 523.25 1046.50 2093.00 4186.01 8372.02];
base_name = {"C" "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
for halftone_number=-57:39
%  printf("%f\n",440*power(2, halftone_number/12));
  frequency(halftone_number-min_note_distance+1) = 440*power(2, halftone_number/12);
end;

for octave_number=0:7
  for halftone_in_octave=1:12
     name(octave_number*12+halftone_in_octave) = sprintf('%s%d',base_name(halftone_in_octave),octave_number);
   end;
end;
name(8*12+1)="C8";

%octave:1> x = 0:200;
%octave:2> plot (x)
%octave:3> set (gca, 'xtick', [0 100 200])
%octave:4> set (gca, 'xticklabel', {'zero', 'one hundred', 'two hundred'})
%octave:5> get (gca, 'xticklabelmode')
%ans = manual 
   


