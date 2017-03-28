#!/usr/bin/perl

# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * Evaluation                          * #

use strict;


my $logo = "Evaluation\n";
print STDERR $logo;

my %param;

GetParameter(@ARGV);

Ssystem("perl NiuTrans.NMT-pure-1best.pl $param{'-1best'} $param{'-1best'}-pure");
Ssystem("perl NiuTrans-generate-xml-for-mteval.pl -1f $param{'-1best'}-pure -tf $param{'-dev'} -rnum $param{'-refn'}");
Ssystem("perl mteval-v13a.pl -r ref.xml -s src.xml -t tst.xml > $param{'-1best'}-pure-bleu");

unlink("$param{'-1best'}-pure.temp");
unlink("ref.xml");
unlink("src.xml");
unlink("tst.xml");

sub GetParameter
{
  if( ( scalar( @_ ) < 6 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "        NiuTrans.NMT-evaluation.pl   [OPTIONS]\n".
                 "[OPTION]\n".
                 "    -1best  :  1best translation\n".
                 "    -dev    :  development format file\n".
                 "    -refn   :  number of reference\n".
                 "[EXAMPLE]\n".
                 "   perl NiuTrans.NMT-evaluation.pl\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }
}


sub Ssystem
{
  print STDERR "Running: @_\n";
  system( @_ );
  if( $? == -1 )
  {
    print STDERR "Error: Failed to execute: @_\n  $!\n";
    exit( 1 );
  }
  elsif( $? & 127 )
  {
    printf STDERR "Error: Execution of: @_\n   die with signal %d, %s coredump\n",
                  ($? & 127 ), ( $? & 128 ) ? 'with' : 'without';
    exit( 1 );
  }
  else
  {
    my $exitcode = $? >> 8;
    print STDERR "Exit code: $exitcode\n" if $exitcode;
    return ! $exitcode;
  }         
}



